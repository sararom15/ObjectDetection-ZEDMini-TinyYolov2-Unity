using System;
using System.IO;
using Barracuda;
using TFClassify;
using UnityEngine;
using System.Linq;
using UnityEngine.UI;
using System.Collections;
using System.Threading.Tasks;
using System.Collections.Generic;
using sl;
using System.Numerics; 
using UnityEngine.Rendering;
using System.Runtime.InteropServices;
using System.Reflection;
using System.Threading;
using System.Diagnostics;




/* Used for debugging purposes. 
 * Attach this script to an object to see data from the camera in the inspector */
public class CameraData : MonoBehaviour
{
    public int ZEDCameraWidth = 2560;
    public int ZEDCameraHeight = 720;

    private static Texture2D boxOutlineTexture;
    private static GUIStyle labelStyle;

    private float cameraScale = 1f;
    private float shiftX = 0f;
    private float shiftY = 0f;
    private float scaleFactor = 1;
    private bool camAvailable;
    
    private bool isWorking = false;
    public Detector detector;

    private IList<BoundingBox> boxOutlines;

    //public RawImage background;
    public AspectRatioFitter fitter;
    public Text uiText;
   
    // ZED variables
    private ZEDManager manager;
    public ZEDCamera Camera_Left;
    public ZEDMat myZedMat;
    public ZEDMat PointCloudMat;
    public ZEDMat depthMat;
    public Texture2D OurTexture;
    public int boxX;
    public int boxY;
    public float objX;
    public float objY;
    public float objZ;


    // Start is called before the first frame update
    void Start()
    {
        // Call the zed manager and define the zed camera
        manager = FindObjectOfType(typeof(ZEDManager)) as ZEDManager;

        Camera_Left = new sl.ZEDCamera(); // Not sure...
        // Retrieves an instance of the camera
        //Camera_Left = sl.ZEDCamera.GetInstance();

        camAvailable = true;

        // Box style
        boxOutlineTexture = new Texture2D(1, 1);
        boxOutlineTexture.SetPixel(0, 0, Color.red);
        boxOutlineTexture.Apply();

        // Label style
        labelStyle = new GUIStyle();
        labelStyle.fontSize = 50;
        labelStyle.normal.textColor = Color.red;

        boxX=0;
        boxY=0;

        //fitter = new AspectRatioFitter();
        //fitter.aspectRatio = (float)Camera_Left.ImageWidth / (float)Camera_Left.ImageHeight;
        //fitter.aspectRatio = ZEDCameraWidth/ZEDCameraHeight;

        // Calculates ratio between the smallest side of the player window in Unity
        // and the desired size for the nn: saves value in scaleFactor (no output)
        CalculateShift(Detector.IMAGE_SIZE);
    }

    // Update is called once per frame
    void Update()
    {
        Stopwatch sw = new Stopwatch();
        sw.Start();

        if (!this.camAvailable)
        {
            return;

        }

        if (! camAvailable){
            print("Camera not available");
            return;
        }
        

        // Define the ZEDMat to save pointer to frame
        myZedMat = new ZEDMat();
        myZedMat.Create((uint)ZEDCameraWidth, (uint)ZEDCameraHeight, ZEDMat.MAT_TYPE.MAT_8U_C4, ZEDMat.MEM.MEM_CPU);   

        // You can't get a texture from the ZED until the ZED is finished initializing. 
        // You can use ZEDManager.IsZEDReady to check this, or subscribe to the ZEDManager.OnZEDReady event.
        //if (!manager.IsZEDReady)
        //    return false;
            
        // Retrieve ZEDCamera image from GPU for access in CPU
        Camera_Left.RetrieveImage(myZedMat, VIEW.LEFT_UNRECTIFIED, ZEDMat.MEM.MEM_CPU);

        // Create matptr to save pointer 
        IntPtr matptr = myZedMat.GetPtr(ZEDMat.MEM.MEM_CPU);
            
        int matwidth = myZedMat.GetWidth();
        int matheight = myZedMat.GetHeight();
        int len = matwidth * matheight * 4;
        byte[] frame_bytes = new byte[len];

        // Marshal copy pointer
        Marshal.Copy(matptr, frame_bytes, 0, len); 
        //OurTexture = new Texture2D(ZEDCameraWidth, ZEDCameraHeight, TextureFormat.BGRA32, false);
        //OurTexture.LoadRawTextureData(frame_bytes);

        // Pixels are flipped?
        
        if (frame_bytes != null) {
            // To try "The image will be upside down. For testing, I just flip the bytes manually with code like this:"
            byte[] flippedbytes = new byte[len];
            int steplength = matwidth * 4;//Two options:  matwidth * 4; / myZedMat.GetStepBytes(); 

            for(int i = 0; i < frame_bytes.Length; i += steplength)
            {
                Array.Copy(frame_bytes, i, flippedbytes, flippedbytes.Length - i - steplength, steplength);
            }
            
            OurTexture = new Texture2D(ZEDCameraWidth, ZEDCameraHeight, TextureFormat.BGRA32, false);
            OurTexture.LoadRawTextureData(flippedbytes);
            //OurTexture.LoadRawTextureData(frame_bytes); // NON FLIPPED
            OurTexture.Apply();

            if (OurTexture==null)
                print("Empty texture");
        }

        //print($"Camera width: {Camera_Left.ImageWidth} and height: {Camera_Left.ImageHeight}");

        /********************** POINT CLOUD **********************/

        // Define the ZEDMat for the point cloud data 
        PointCloudMat = new ZEDMat();
        PointCloudMat.Create((uint)ZEDCameraWidth, (uint)ZEDCameraHeight, ZEDMat.MAT_TYPE.MAT_32F_C4, ZEDMat.MEM.MEM_CPU);
        Camera_Left.RetrieveMeasure(PointCloudMat, MEASURE.XYZRGBA, ZEDMat.MEM.MEM_CPU);
        float4 point3D;
        PointCloudMat.GetValue(boxX, boxY, out point3D, ZEDMat.MEM.MEM_CPU); 
        objX = point3D.r;
        objY = point3D.g;
        objZ = point3D.b;
        //print($"Point cloud results: z={objZ}; x={objX}; y={objY}");
        
        /********************** DEPTH MAP **********************/
        
        depthMat = new ZEDMat();
        depthMat.Create((uint)ZEDCameraWidth, (uint)ZEDCameraHeight, ZEDMat.MAT_TYPE.MAT_32F_C1, ZEDMat.MEM.MEM_CPU);
        Camera_Left.RetrieveMeasure(depthMat, MEASURE.DEPTH, ZEDMat.MEM.MEM_CPU);
        depthMat.GetValue(boxX, boxY, out float depth_value, ZEDMat.MEM.MEM_CPU);
        //print($"Depth map results: z={depth_value}");

        
        // Call detect
        TFDetect();

        
        sw.Stop();
        TimeSpan ts = sw.Elapsed;
        string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds / 10);
        print($"RunTime: {elapsedTime}");

    }


    // Calls DrawBoxOutline to draw boxes: passes the results of the detector and
    // draws both the rectangle and the label
    public void OnGUI()
    {
        //DrawLabel(new Rect(1 + 10, 2 + 10, 200, 20), $"E' ORA DELLO SPRITZ");
        
        if (this.boxOutlines != null && this.boxOutlines.Any())
        {
            DrawBoxOutline(boxOutlines[0], scaleFactor, shiftX, shiftY);
            
            // for(int i = 1; i < 3; i += 1)
            // {
            //     if (boxOutlines[i].Confidence>=30)
            //         DrawBoxOutline(boxOutlines[i], scaleFactor, shiftX, shiftY);
            // }
            
            //print($"Label is {boxOutlines[0].Label}, confidence is {boxOutlines[0].Confidence}");
            
            /**
            foreach (var outline in this.boxOutlines)
            {
                DrawBoxOutline(outline, scaleFactor, shiftX, shiftY);
            }
            **/
        }

        // Added to remove the GUI error! *new*
        GUIUtility.ExitGUI(); 
    }


    // Returns the ratio between the smallest side of the player window in Unity
    // and the desired size for the nn 
    private void CalculateShift(int inputSize)
    {
        int smallest;

        if (Screen.width < Screen.height)
        {
            smallest = Screen.width;
            this.shiftY = (Screen.height - smallest) / 2f;
        }
        else
        {
            smallest = Screen.height;
            this.shiftX = (Screen.width - smallest) / 2f;
        }

        this.scaleFactor = smallest / (float)inputSize;
    }


    private void TFDetect()
    {
        if (this.isWorking)
        {
            return;
        }
        
        

        this.isWorking = true;
        StartCoroutine(ProcessImage(Detector.IMAGE_SIZE, result =>
        {
            StartCoroutine(this.detector.Detect(result, boxes =>
            {
                this.boxOutlines = boxes;
                Resources.UnloadUnusedAssets();
                this.isWorking = false;
            }));
        }));
        


    }


    //Retrieve Image


    // Calls crop and square function (in TextureTools) by giving the right scale and
    // rotation values desired for the camera
    private IEnumerator ProcessImage(int inputSize, System.Action<Color32[]> callback)
    {
        // Crop square from the center of the image
        yield return StartCoroutine(TextureTools.CropSquare(OurTexture, TextureTools.RectOptions.Center, snap =>
            {
                var scaled = Scale(snap, inputSize);
                var rotated = Rotate(scaled.GetPixels32(), scaled.width, scaled.height);
                callback(rotated);
            }));
    }

    private void DrawBoxOutline(BoundingBox outline, float scaleFactor, float shiftX, float shiftY)
    {
        var x = outline.Dimensions.X * scaleFactor + shiftX;
        var width = outline.Dimensions.Width * scaleFactor;
        var y = outline.Dimensions.Y * scaleFactor + shiftY+200;
        var height = outline.Dimensions.Height * scaleFactor;

        boxX=(int)(x+(width/2));
        boxY=(int)(y+(height/2));

        DrawRectangle(new Rect(x, y+shiftY, width, height), 4, Color.red);
        labelStyle.fontSize=25;
        DrawLabel(new Rect(x + 5 , y + height + 5, 200, 20), $"{outline.Label}: {(int)(outline.Confidence * 100)}%, x={Math.Round(objX,2)}, y={Math.Round(objY,2)}, z={Math.Round(objZ,2)}", labelStyle);
        

        labelStyle.fontSize=20;
        DrawLabel(new Rect(x , y, 200, 20), $"{outline.Label}, x={Math.Round(objX,2)},\ny={Math.Round(objY,2)},\nz={Math.Round(objZ,2)}", labelStyle);

    }


    public static void DrawRectangle(Rect area, int frameWidth, Color color)
    {
        Rect lineArea = area;
        lineArea.height = frameWidth;
        GUI.DrawTexture(lineArea, boxOutlineTexture); // Top line

        lineArea.y = area.yMax - frameWidth;
        GUI.DrawTexture(lineArea, boxOutlineTexture); // Bottom line

        lineArea = area;
        lineArea.width = frameWidth;
        GUI.DrawTexture(lineArea, boxOutlineTexture); // Left line

        lineArea.x = area.xMax - frameWidth;
        GUI.DrawTexture(lineArea, boxOutlineTexture); // Right line
    }




    private static void DrawLabel(Rect position, string text, GUIStyle labelStyle)
    {
        GUI.Label(position, text, labelStyle);
    }

    private Texture2D Scale(Texture2D texture, int imageSize) //Returns a scaled copy of given texture
    {
        var scaled = TextureTools.scaled(texture, imageSize, imageSize, FilterMode.Bilinear);

        return scaled;
    }


    private Color32[] Rotate(Color32[] pixels, int width, int height)
    {
        return TextureTools.RotateImageMatrix(
                pixels, width, height, 0);
    }



 

}