using System;
using Barracuda;
using UnityEngine;
using Debug = UnityEngine.Debug;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Diagnostics;


public class Detector : MonoBehaviour
{   // These need to be set on Unity user interface
    public NNModel modelFile;
    public TextAsset labelsFile;

    // for .onnx model
    private const int IMAGE_MEAN = 0; // How do we choose it?.....................................
    private const float IMAGE_STD = 1f; // How do we choose it?.....................................
    private const string INPUT_NAME = "image";
    private const string OUTPUT_NAME = "grid";
    
    // Tiny Yolo v2 has input size of 416x416 3channels image
    public const int IMAGE_SIZE = 416;

    private const float IOU_desiredThreshold = 0.3f;

    private IWorker worker;

    // Tiny Yolo v2 has output size of 125x13x13  
    // It recognizes 20 classes
    public const int ROW_COUNT = 13;
    public const int COL_COUNT = 13;
    public const int BOXES_PER_CELL = 5;
    public const int BOX_INFO_FEATURE_COUNT = 5;
    public const int CLASS_COUNT = 20;
    public const float CELL_WIDTH = 32; 
    public const float CELL_HEIGHT = 32;
    private string[] labels;

    private float[] anchors = new float[]
    {
        1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
    };
    
    // Reads labels, split them to be readable
    // Loads ONNX file
    // Creates worker
    public void Start()
    {
        this.labels = Regex.Split(this.labelsFile.text, "\n|\r|\r\n")
            .Where(s => !String.IsNullOrEmpty(s)).ToArray();
        var model = ModelLoader.Load(this.modelFile);
        this.worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
    }

    // Object detection main: 
    // Modifies input image by inputs=TransformInput (subtract mean, divide for variance)
    // Creates the tensor with a dictionary of name-tensor pairs: image-inputs (WHY IMAGE?...............)
    // Executs model (async)
    // Reads output
    public IEnumerator Detect(Color32[] picture, System.Action<IList<BoundingBox>> callback)
    {

        
        using (var tensor = TransformInput(picture, IMAGE_SIZE, IMAGE_SIZE))
        {
            var inputs = new Dictionary<string, Tensor>();
            inputs.Add(INPUT_NAME, tensor);
            yield return StartCoroutine(worker.ExecuteAsync(inputs));

            var output = worker.PeekOutput(OUTPUT_NAME);
            // results are the bounding boxes and info
            var results = ParseOutputs(output); 
            var boxes = FilterBoundingBoxes(results, 5, IOU_desiredThreshold);

            callback(boxes);
        }


    }

    // For each picture frame transform pixels (Note: PREPROCESSING DONE IN PHONECAMERA):
    // Subtract mean, divide for variance
    public static Tensor TransformInput(Color32[] pic, int width, int height)
    {
        float[] floatValues = new float[width * height * 3];

        for (int i = 0; i < pic.Length; ++i)
        {
            var color = pic[i];

            floatValues[i * 3 + 0] = (color.r - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 1] = (color.g - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 2] = (color.b - IMAGE_MEAN) / IMAGE_STD;
        }

        return new Tensor(1, height, width, 3, floatValues);
    }

    // Handles the yolo outputs
    private IList<BoundingBox> ParseOutputs(Tensor yoloModelOutput, float threshold = 0.1F)//.3F) // How do I choose threshold?.....................
    {
        var boxes = new List<BoundingBox>();

        for (int cy = 0; cy < COL_COUNT; cy++) // Cycle though columns of grid cell: 13
        {
            for (int cx = 0; cx < ROW_COUNT; cx++) // Cycle through rows of grid cell: 13
            {
                for (int box = 0; box < BOXES_PER_CELL; box++) // Cycle through predicted bounding boxes
                {   // CLASS_COUNT + BOX_INFO_FEATURE_COUNT= 20 + 5 = 25 (data elements that describe each bounding box)
                    var channel = (box * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT)); 
                    // Extracts x,y,w,h bb info from the output
                    var bbd = ExtractBoundingBoxDimensions(yoloModelOutput, cx, cy, channel);
                    // Extracts confidence info in a range [0,1]
                    float confidence = GetConfidence(yoloModelOutput, cx, cy, channel);

                    if (confidence < threshold)
                    {
                        continue;
                    }
                    //Extracts classes probabilities in a range [0,1]
                    float[] predictedClasses = ExtractClasses(yoloModelOutput, cx, cy, channel);
                    // Computes highest scoring class
                    var (topResultIndex, topResultScore) = GetTopResult(predictedClasses);
                    var topScore = topResultScore * confidence;

                    if (topScore < threshold)
                    {
                        continue;
                    }

                    var mappedBoundingBox = MapBoundingBoxToCell(cx, cy, box, bbd);

                    //print($"Label is {labels[topResultIndex]} at {Time.time} in detect");
                    
                    // Create the array of the final bounding boxes: each box contains dimension, confidence, label
                    boxes.Add(new BoundingBox
                    {
                        Dimensions = new BoundingBoxDimensions
                        {
                            X = (mappedBoundingBox.X - mappedBoundingBox.Width / 2),
                            Y = (mappedBoundingBox.Y - mappedBoundingBox.Height / 2),
                            Width = mappedBoundingBox.Width,
                            Height = mappedBoundingBox.Height,
                        },
                        Confidence = topScore,
                        Label = labels[topResultIndex] 
                    });
                }
            }
        }

        return boxes;
    }


    private float Sigmoid(float value)
    {
        var k = (float)Math.Exp(value);

        return k / (1.0f + k);
    }


    private float[] Softmax(float[] values)
    {
        var maxVal = values.Max();
        var exp = values.Select(v => Math.Exp(v - maxVal));
        var sumExp = exp.Sum();

        return exp.Select(v => (float)(v / sumExp)).ToArray();
    }

    // Extracts information of bounding boxes depending on the format of tinyyolov2
    private BoundingBoxDimensions ExtractBoundingBoxDimensions(Tensor modelOutput, int x, int y, int channel)
    {
        return new BoundingBoxDimensions
        {
            X = modelOutput[0, x, y, channel],
            Y = modelOutput[0, x, y, channel + 1],
            Width = modelOutput[0, x, y, channel + 2],
            Height = modelOutput[0, x, y, channel + 3]
        };
    }

    // Extracts confidence, and applies sigmoidal function to have it in [0, 1] range
    private float GetConfidence(Tensor modelOutput, int x, int y, int channel)
    {
        return Sigmoid(modelOutput[0, x, y, channel + 4]);
    }

    // Map the boxes to the cells
    private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
    {
        return new CellDimensions
        {
            X = ((float)y + Sigmoid(boxDimensions.X)) * CELL_WIDTH,
            Y = ((float)x + Sigmoid(boxDimensions.Y)) * CELL_HEIGHT,
            Width = (float)Math.Exp(boxDimensions.Width) * CELL_WIDTH * anchors[box * 2],
            Height = (float)Math.Exp(boxDimensions.Height) * CELL_HEIGHT * anchors[box * 2 + 1],
        };
    }

    // Extracts classes probabilities, and applies a softmax function to have them normalized
    public float[] ExtractClasses(Tensor modelOutput, int x, int y, int channel)
    {   // create 1x20 array of classes
        float[] predictedClasses = new float[CLASS_COUNT];
        int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;

        for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
        {
            predictedClasses[predictedClass] = modelOutput[0, x, y, predictedClass + predictedClassOffset];
        }

        return Softmax(predictedClasses);
    }

    // Returns the score and index of the highest probability class
    private ValueTuple<int, float> GetTopResult(float[] predictedClasses)
    {
        return predictedClasses
            .Select((predictedClass, index) => (Index: index, Value: predictedClass))
            .OrderByDescending(result => result.Value)
            .First();
    }

    // Computes the area in common between the two input boxes and divides it by the union of the two
    private float IntersectionOverUnion(Rect boundingBoxA, Rect boundingBoxB)
    {
        var areaA = boundingBoxA.width * boundingBoxA.height;

        if (areaA <= 0)
            return 0;

        var areaB = boundingBoxB.width * boundingBoxB.height;

        if (areaB <= 0)
            return 0;

        var minX = Math.Max(boundingBoxA.xMin, boundingBoxB.xMin);
        var minY = Math.Max(boundingBoxA.yMin, boundingBoxB.yMin);
        var maxX = Math.Min(boundingBoxA.xMax, boundingBoxB.xMax);
        var maxY = Math.Min(boundingBoxA.yMax, boundingBoxB.yMax);

        var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    // Compares bounding boxes in a similar area, keeps only the ones that have highest confidence
    private IList<BoundingBox> FilterBoundingBoxes(IList<BoundingBox> boxes, int limit, float IOU_threshold)
    {
        var activeCount = boxes.Count;
        var isActiveBoxes = new bool[boxes.Count];

        // Set all boxes to active
        for (int i = 0; i < isActiveBoxes.Length; i++)
        {
            isActiveBoxes[i] = true;
        }

        // Order by descending confidence
        var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                .OrderByDescending(b => b.Box.Confidence)
                .ToList();

        var results = new List<BoundingBox>();

        // Cycle through ordered boxes, if two boxes cover a similar area, keep only the one with highest confidence
        for (int i = 0; i < boxes.Count; i++)
        {
            if (isActiveBoxes[i])
            {   // I take active box A with index i
                var boxA = sortedBoxes[i].Box;
                results.Add(boxA);

                if (results.Count >= limit)
                    break;

                for (var j = i + 1; j < boxes.Count; j++)
                {
                    if (isActiveBoxes[j])
                    {   // I take active box B at index i+1
                        var boxB = sortedBoxes[j].Box;

                        if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > IOU_threshold)
                        {   // If the ratio intersection/union > threshold: deactivate B
                            isActiveBoxes[j] = false;
                            activeCount--;

                            if (activeCount <= 0)
                                break;
                        }
                    }
                }

                if (activeCount <= 0)
                    break;
            }
        }

        return results;
    }
}


public class DimensionsBase
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Height { get; set; }
    public float Width { get; set; }
}


public class BoundingBoxDimensions : DimensionsBase { }

class CellDimensions : DimensionsBase { }


public class BoundingBox
{
    public BoundingBoxDimensions Dimensions { get; set; }

    public string Label { get; set; }

    public float Confidence { get; set; }

    public Rect Rect
    {
        get { return new Rect(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
    }

    public override string ToString()
    {
        return $"{Label}:{Confidence}, {Dimensions.X}:{Dimensions.Y} - {Dimensions.Width}:{Dimensions.Height}";
    }
}