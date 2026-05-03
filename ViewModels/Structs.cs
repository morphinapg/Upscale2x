using Avalonia.Media.Imaging;
using ComputeSharp;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Upscale2x.ViewModels
{

    public struct InputPixelData
    {
        public float4 Pixel1;
        public float4 Pixel2;
        public float4 Pixel3;
        public float4 Pixel4;
        public float4 Pixel5;
        public float4 Pixel6;
        public float4 Pixel7;
        public float4 Pixel8;
        public float4 Pixel9;
        public float4 Pixel10;
        public float4 Pixel11;
        public float4 Pixel12;
        public float4 Pixel13;
        public float4 Pixel14;
        public float4 Pixel15;
        public float4 Pixel16;
        public float4 Pixel17;
        public float4 Pixel18;
        public float4 Pixel19;
        public float4 Pixel20;
        public float4 Pixel21;
        public float4 Pixel22;
        public float4 Pixel23;
        public float4 Pixel24;
        public float4 Pixel25;
    }

    public struct Layer1Data
    {
        public float4 Neuron1;
        public float4 Neuron2;
        public float4 Neuron3;
        public float4 Neuron4;
        public float4 Neuron5;
        public float4 Neuron6;
        public float4 Neuron7;
        public float4 Neuron8;
        public float4 Neuron9;
        public float4 Neuron10;
        public float4 Neuron11;
        public float4 Neuron12;
        public float4 Neuron13;
        public float4 Neuron14;
        public float4 Neuron15;
        public float4 Neuron16;
        public float4 Neuron17;
        public float4 Neuron18;
        public float4 Neuron19;
        public float4 Neuron20;
        public float4 Neuron21;
    }

    public struct Layer2Data
    {
        public float4 Neuron1;
        public float4 Neuron2;
        public float4 Neuron3;
        public float4 Neuron4;
        public float4 Neuron5;
        public float4 Neuron6;
        public float4 Neuron7;
        public float4 Neuron8;
        public float4 Neuron9;
        public float4 Neuron10;
    }

    //public struct Layer3Data
    //{
    //    public float4 Neuron1;
    //    public float4 Neuron2;
    //    public float4 Neuron3;
    //}

    [ThreadGroupSize(DefaultThreadGroupSizes.XY)]
    [GeneratedComputeShaderDescriptor]
    public readonly partial struct CopyToFloat(ReadWriteTexture2D<Rgba64, float4> InputImage, ReadWriteTexture2D<float4> OutputImage) : IComputeShader
    {
        public void Execute()
        {
            OutputImage[ThreadIds.XY] = InputImage[ThreadIds.XY];
        }
    }

    [ThreadGroupSize(DefaultThreadGroupSizes.XY)]
    [GeneratedComputeShaderDescriptor]
    public readonly partial struct GPUDownscale(ReadWriteTexture2D<Rgba64, float4> InputImage) : IComputeShader<float4>
    {
        public float4 Execute()
        {
            int X = ThreadIds.X * 2;
            int Y = ThreadIds.Y * 2;

            float4
                p1 = InputImage[X, Y],
                p2 = InputImage[X + 1, Y],
                p3 = InputImage[X, Y + 1],
                p4 = InputImage[X + 1, Y + 1];

            return (p1 + p2 + p3 + p4) / 4f;
        }
    }

    public struct ModelParameters
    {
        public float[] InputAverages;
        public float[] InputDeviations;
        public float[,] Layer1Weights;
        public float[] Layer1Biases;
        public float[] Layer1OutputBiases;
        public float[,] Layer2Weights;
        public float[] Layer2Biases;
        public float[] Layer2OutputBiases;
        public float[,] Layer3Weights;
        public float[] Layer3Biases;
        public float[] Layer3OutputBiases; 
        public float[,] OutputLayerWeights;
        public float[] OutputLayerBiases;
    }

    [ThreadGroupSize(DefaultThreadGroupSizes.XYZ)]
    [GeneratedComputeShaderDescriptor]
    public readonly partial struct Upscale_GPU(
            ReadWriteTexture2D<Rgba64, float4> ImageToUpscale,
            ReadWriteTexture2D<Rgba64, float4> OriginalImage,
            ReadWriteTexture3D<float> CalculatedErrors,
            ReadWriteTexture2D<float4> OutputTexture,
            bool CalculateErrors,
            bool Use2DOutput,
            bool Accelerated,
            bool Analyze,
            ReadOnlyTexture2D<float> InputAverages,
            ReadOnlyTexture2D<float> InputDeviations,
            ReadOnlyTexture3D<float> Layer1Weights,
            ReadOnlyTexture2D<float> Layer1Biases,
            ReadOnlyTexture2D<float> Layer1OutputBiases,
            ReadOnlyTexture3D<float> Layer2Weights,
            ReadOnlyTexture2D<float> Layer2Biases,
            ReadOnlyTexture2D<float> Layer2OutputBiases,
            //ReadOnlyTexture3D<float> Layer3Weights,
            //ReadOnlyTexture2D<float> Layer3Biases,
            //ReadOnlyTexture2D<float> Layer3OutputBiases,
            ReadOnlyTexture3D<float> OutputLayerWeights,
            ReadOnlyTexture2D<float> OutputLayerBiases
                ) : IComputeShader
    {
        public void Execute()
        {
            //First, determine pixel coordinates for sampling

            int
                X1 = ThreadIds.X - 2,
                X2 = ThreadIds.X - 1,
                X3 = ThreadIds.X,
                X4 = ThreadIds.X + 1,
                X5 = ThreadIds.X + 2,
                Y1 = ThreadIds.Y - 2,
                Y2 = ThreadIds.Y - 1,
                Y3 = ThreadIds.Y,
                Y4 = ThreadIds.Y + 1,
                Y5 = ThreadIds.Y + 2,
                Z = ThreadIds.Z,
                MaxWidth = ImageToUpscale.Width - 1,
                MaxHeight = ImageToUpscale.Height - 1;

            //Now, apply mirroring if any coordinates are out of range            

            if (X1 < 0)
                X1 = 0 - X1;
            if (X2 < 0)
                X2 = 0 - X2;
            if (X4 > MaxWidth)
                X4 = MaxWidth - (X4 - MaxWidth);
            if (X5 > MaxWidth)
                X5 = MaxWidth - (X5 - MaxWidth);
            if (Y1 < 0)
                Y1 = 0 - Y1;
            if (Y2 < 0)
                Y2 = 0 - Y2;
            if (Y4 > MaxHeight)
                Y4 = MaxHeight - (Y4 - MaxHeight);
            if (Y5 > MaxHeight)
                Y5 = MaxHeight - (Y5 - MaxHeight);

            //Now, sample and normalize input pixels

            InputPixelData InputPixels = default;

            //Reformat pixels to be relative to Pixel13
            float4 
                Pixel1 = ImageToUpscale[X1, Y1],
                Pixel2 = ImageToUpscale[X2, Y1],
                Pixel3 = ImageToUpscale[X3, Y1],
                Pixel4 = ImageToUpscale[X4, Y1],
                Pixel5 = ImageToUpscale[X5, Y1],
                Pixel6 = ImageToUpscale[X1, Y2],
                Pixel7 = ImageToUpscale[X2, Y2],
                Pixel8 = ImageToUpscale[X3, Y2],
                Pixel9 = ImageToUpscale[X4, Y2],
                Pixel10 = ImageToUpscale[X5, Y2],
                Pixel11 = ImageToUpscale[X1, Y3],
                Pixel12 = ImageToUpscale[X2, Y3],
                ReferencePixel = ImageToUpscale[X3, Y3],
                Pixel14 = ImageToUpscale[X4, Y3],
                Pixel15 = ImageToUpscale[X5, Y3],
                Pixel16 = ImageToUpscale[X1, Y4],
                Pixel17 = ImageToUpscale[X2, Y4],
                Pixel18 = ImageToUpscale[X3, Y4],
                Pixel19 = ImageToUpscale[X4, Y4],
                Pixel20 = ImageToUpscale[X5, Y4],
                Pixel21 = ImageToUpscale[X1, Y5],
                Pixel22 = ImageToUpscale[X2, Y5],
                Pixel23 = ImageToUpscale[X3, Y5],
                Pixel24 = ImageToUpscale[X4, Y5],
                Pixel25 = ImageToUpscale[X5, Y5];

                InputPixels.Pixel1 = ((Pixel1 - ReferencePixel) - InputAverages[Z, 0]) / InputDeviations[Z, 0];
                InputPixels.Pixel2 = ((Pixel2 - ReferencePixel) - InputAverages[Z, 1]) / InputDeviations[Z, 1];
                InputPixels.Pixel3 = ((Pixel3 - ReferencePixel) - InputAverages[Z, 2]) / InputDeviations[Z, 2];
                InputPixels.Pixel4 = ((Pixel4 - ReferencePixel) - InputAverages[Z, 3]) / InputDeviations[Z, 3];
                InputPixels.Pixel5 = ((Pixel5 - ReferencePixel) - InputAverages[Z, 4]) / InputDeviations[Z, 4];
                InputPixels.Pixel6 = ((Pixel6 - ReferencePixel) - InputAverages[Z, 5]) / InputDeviations[Z, 5];
                InputPixels.Pixel7 = ((Pixel7 - ReferencePixel) - InputAverages[Z, 6]) / InputDeviations[Z, 6];
                InputPixels.Pixel8 = ((Pixel8 - ReferencePixel) - InputAverages[Z, 7]) / InputDeviations[Z, 7];
                InputPixels.Pixel9 = ((Pixel9 - ReferencePixel) - InputAverages[Z, 8]) / InputDeviations[Z, 8];
                InputPixels.Pixel10 = ((Pixel10 - ReferencePixel) - InputAverages[Z, 9]) / InputDeviations[Z, 9];
                InputPixels.Pixel11 = ((Pixel11 - ReferencePixel) - InputAverages[Z, 10]) / InputDeviations[Z, 10];
                InputPixels.Pixel12 = ((Pixel12 - ReferencePixel) - InputAverages[Z, 11]) / InputDeviations[Z, 11];
                InputPixels.Pixel13 = (0f - InputAverages[Z, 12]) / InputDeviations[Z, 12];
                InputPixels.Pixel14 = ((Pixel14 - ReferencePixel) - InputAverages[Z, 13]) / InputDeviations[Z, 13];
                InputPixels.Pixel15 = ((Pixel15 - ReferencePixel) - InputAverages[Z, 14]) / InputDeviations[Z, 14];
                InputPixels.Pixel16 = ((Pixel16 - ReferencePixel) - InputAverages[Z, 15]) / InputDeviations[Z, 15];
                InputPixels.Pixel17 = ((Pixel17 - ReferencePixel) - InputAverages[Z, 16]) / InputDeviations[Z, 16];
                InputPixels.Pixel18 = ((Pixel18 - ReferencePixel) - InputAverages[Z, 17]) / InputDeviations[Z, 17];
                InputPixels.Pixel19 = ((Pixel19 - ReferencePixel) - InputAverages[Z, 18]) / InputDeviations[Z, 18];
                InputPixels.Pixel20 = ((Pixel20 - ReferencePixel) - InputAverages[Z, 19]) / InputDeviations[Z, 19];
                InputPixels.Pixel21 = ((Pixel21 - ReferencePixel) - ReferencePixel - InputAverages[Z, 20]) / InputDeviations[Z, 20];
                InputPixels.Pixel22 = ((Pixel22 - ReferencePixel) - InputAverages[Z, 21]) / InputDeviations[Z, 21];
                InputPixels.Pixel23 = ((Pixel23 - ReferencePixel) - InputAverages[Z, 22]) / InputDeviations[Z, 22];
                InputPixels.Pixel24 = ((Pixel24 - ReferencePixel) - InputAverages[Z, 23]) / InputDeviations[Z, 23];
                InputPixels.Pixel25 = ((Pixel25 - ReferencePixel) - InputAverages[Z, 24]) / InputDeviations[Z, 24];

            
            float
                CenterWeight = 0.5625f,
                CloseCornerWeight = 0.1875f,
                FarCornerWeight = 0.0625f;

            float4
                Reference1 = Pixel7 * FarCornerWeight + Pixel8 * CloseCornerWeight + Pixel12 * CloseCornerWeight + ReferencePixel * CenterWeight,
                Reference2 = Pixel8 * CloseCornerWeight + Pixel9 * FarCornerWeight + ReferencePixel * CenterWeight + Pixel14 * CloseCornerWeight,
                Reference3 = Pixel12 * CloseCornerWeight + ReferencePixel * CenterWeight + Pixel17 * FarCornerWeight + Pixel18 * CloseCornerWeight,
                Reference4 = ReferencePixel * CenterWeight + Pixel14 * CloseCornerWeight + Pixel18 * CloseCornerWeight + Pixel19 * FarCornerWeight;

            //iterate through each output pixel, calculating the necessary layer outputs and final output for each
            Layer1Data Layer1 = default;
            Layer2Data Layer2 = default;
            //Layer3Data Layer3 = default;
            //Float4x4 Outputs = default;

            //for (int i = 0; i < 4; i++)
            //{
                //Calculate Layer 1 outputs
                Layer1.Neuron1 = GetLayer1Output(0, InputPixels, Z);
                Layer1.Neuron2 = GetLayer1Output(1, InputPixels, Z);
                Layer1.Neuron3 = GetLayer1Output(2, InputPixels, Z);
                Layer1.Neuron4 = GetLayer1Output(3, InputPixels, Z);
                Layer1.Neuron5 = GetLayer1Output(4, InputPixels, Z);
                Layer1.Neuron6 = GetLayer1Output(5, InputPixels, Z);
                Layer1.Neuron7 = GetLayer1Output(6, InputPixels, Z);
                Layer1.Neuron8 = GetLayer1Output(7, InputPixels, Z);
                Layer1.Neuron9 = GetLayer1Output(8, InputPixels, Z);
                Layer1.Neuron10 = GetLayer1Output(9, InputPixels, Z);
                Layer1.Neuron11 = GetLayer1Output(10, InputPixels, Z);
                Layer1.Neuron12 = GetLayer1Output(11, InputPixels, Z);
                Layer1.Neuron13 = GetLayer1Output(12, InputPixels, Z);
                Layer1.Neuron14 = GetLayer1Output(13, InputPixels, Z);
                Layer1.Neuron15 = GetLayer1Output(14, InputPixels, Z);
                Layer1.Neuron16 = GetLayer1Output(15, InputPixels, Z);
                Layer1.Neuron17 = GetLayer1Output(16, InputPixels, Z);
                Layer1.Neuron18 = GetLayer1Output(17, InputPixels, Z);


                //Calculate Layer 2 outputs
                Layer2.Neuron1 = GetLayer2Output(0, Layer1, Z);
                Layer2.Neuron2 = GetLayer2Output(1, Layer1, Z);
                Layer2.Neuron3 = GetLayer2Output(2, Layer1, Z);
                Layer2.Neuron4 = GetLayer2Output(3, Layer1, Z);
                Layer2.Neuron5 = GetLayer2Output(4, Layer1, Z);
                Layer2.Neuron6 = GetLayer2Output(5, Layer1, Z);
                Layer2.Neuron7 = GetLayer2Output(6, Layer1, Z);
                Layer2.Neuron8 = GetLayer2Output(7, Layer1, Z);
                Layer2.Neuron9 = GetLayer2Output(8, Layer1, Z);

                ////Calculate Layer 3 outputs
                //Layer3.Neuron1 = GetLayer3Output(0, Layer2, Z);
                //Layer3.Neuron2 = GetLayer3Output(1, Layer2, Z);
                //Layer3.Neuron3 = GetLayer3Output(2, Layer2, Z);

                ////Calculate Output Pixel
                //Outputs[i] = GetOutputPixel(i, Layer3, Z);
            //}

            float4
                Output1 = GetOutputPixel(0, Layer2, Z),
                Output2 = GetOutputPixel(1, Layer2, Z),
                Output3 = GetOutputPixel(2, Layer2, Z),
                Output4 = GetOutputPixel(3, Layer2, Z);

            if (Accelerated)
            {                   

                Output1 += Reference1;
                Output2 += Reference2;
                Output3 += Reference3;
                Output4 += Reference4;
            }
            else
            {
                Output1 += ReferencePixel;
                Output2 += ReferencePixel;
                Output3 += ReferencePixel;
                Output4 += ReferencePixel;
            }


            //Calculate Output coordinates

            int2
                XY1 = new Int2(ThreadIds.X * 2, ThreadIds.Y * 2),
                XY2 = new Int2(XY1.X + 1, XY1.Y),
                XY3 = new Int2(XY1.X, XY1.Y + 1),
                XY4 = new Int2(XY1.X + 1, XY1.Y + 1);

            int3
                XYZ1 = new int3(XY1.X, XY1.Y, Z),
                XYZ2 = new Int3(XY2.X, XY2.Y, Z),
                XYZ3 = new Int3(XY3.X, XY3.Y, Z),
                XYZ4 = new int3(XY4.X, XY4.Y, Z);
            

            if (ReferencePixel.A == 1f || ReferencePixel.A == 0f)
            {
                Output1.A = ReferencePixel.A;
                Output2.A = ReferencePixel.A;
                Output3.A = ReferencePixel.A;
                Output4.A = ReferencePixel.A;
            }

            

            //Calculate errors if necessary, then store output

            if (CalculateErrors)
            {
                Output1 = Hlsl.Abs(Output1 - OriginalImage[XY1]);
                Output2 = Hlsl.Abs(Output2 - OriginalImage[XY2]) ;
                Output3 = Hlsl.Abs(Output3 - OriginalImage[XY3]);
                Output4 = Hlsl.Abs(Output4 - OriginalImage[XY4]) ;

                Output1.R = Hlsl.Pow(Output1.R + Output1.G + Output1.B + Output1.A, 2);
                Output1.G = 0f;
                Output1.B = 0f;
                Output1.A = 0f;

                Output2.R = Hlsl.Pow(Output2.R + Output2.G + Output2.B + Output2.A, 2);
                Output2.G = 0f;
                Output2.B = 0f;
                Output2.A = 0f;

                Output3.R = Hlsl.Pow(Output3.R + Output3.G + Output3.B + Output3.A, 2);
                Output3.G = 0f;
                Output3.B = 0f;
                Output3.A = 0f;

                Output4.R = Hlsl.Pow(Output4.R + Output4.G + Output4.B + Output4.A, 2);
                Output4.G = 0f;
                Output4.B = 0f;
                Output4.A = 0f;


                if (Use2DOutput)
                {
                    OutputTexture[XY1] = Output1;
                    OutputTexture[XY2] = Output2;
                    OutputTexture[XY3] = Output3;
                    OutputTexture[XY4] = Output4;
                }
                else
                {
                    CalculatedErrors[XYZ1] = Output1.R;
                    CalculatedErrors[XYZ2] = Output2.R;
                    CalculatedErrors[XYZ3] = Output3.R;
                    CalculatedErrors[XYZ4] = Output4.R;
                }
            }
            else
            {
                if (Analyze)
                {
                    Output1 = Output1 - Reference1 + 0.5f;                    
                    Output2 = Output2 - Reference2 + 0.5f;
                    Output3 = Output3 - Reference3 + 0.5f;
                    Output4 = Output4 - Reference4 + 0.5f;

                    Output1 += (Output1.A - 0.5f);
                    Output1.A = 1;

                    Output2 += (Output2.A - 0.5f);
                    Output2.A = 1;

                    Output3 += (Output3.A - 0.5f);
                    Output3.A = 1;

                    Output4 += (Output4.A - 0.5f);
                    Output4.A = 1;
                }


                OutputTexture[XY1] = Output1;
                OutputTexture[XY2] = Output2;
                OutputTexture[XY3] = Output3;
                OutputTexture[XY4] = Output4;
            }

            

        }

        float4 Sigmoid(float4 x) => 1 / (1 + Hlsl.Exp(-x));

        float4 GetLayer1Output(int Neuron, InputPixelData Pixels, int Z)
        {
            //Neuron = 18 * Pixel + Neuron;

            float4 value = default;
            value += Layer1Weights[Z, 0, Neuron] * Pixels.Pixel1;
            value += Layer1Weights[Z, 1, Neuron] * Pixels.Pixel2;
            value += Layer1Weights[Z, 2, Neuron] * Pixels.Pixel3;
            value += Layer1Weights[Z, 3, Neuron] * Pixels.Pixel4;
            value += Layer1Weights[Z, 4, Neuron] * Pixels.Pixel5;
            value += Layer1Weights[Z, 5, Neuron] * Pixels.Pixel6;
            value += Layer1Weights[Z, 6, Neuron] * Pixels.Pixel7;
            value += Layer1Weights[Z, 7, Neuron] * Pixels.Pixel8;
            value += Layer1Weights[Z, 8, Neuron] * Pixels.Pixel9;
            value += Layer1Weights[Z, 9, Neuron] * Pixels.Pixel10;
            value += Layer1Weights[Z, 10, Neuron] * Pixels.Pixel11;
            value += Layer1Weights[Z, 11, Neuron] * Pixels.Pixel12;
            value += Layer1Weights[Z, 12, Neuron] * Pixels.Pixel13;
            value += Layer1Weights[Z, 13, Neuron] * Pixels.Pixel14;
            value += Layer1Weights[Z, 14, Neuron] * Pixels.Pixel15;
            value += Layer1Weights[Z, 15, Neuron] * Pixels.Pixel16;
            value += Layer1Weights[Z, 16, Neuron] * Pixels.Pixel17;
            value += Layer1Weights[Z, 17, Neuron] * Pixels.Pixel18;
            value += Layer1Weights[Z, 18, Neuron] * Pixels.Pixel19;
            value += Layer1Weights[Z, 19, Neuron] * Pixels.Pixel20;
            value += Layer1Weights[Z, 20, Neuron] * Pixels.Pixel21;
            value += Layer1Weights[Z, 21, Neuron] * Pixels.Pixel22;
            value += Layer1Weights[Z, 22, Neuron] * Pixels.Pixel23;
            value += Layer1Weights[Z, 23, Neuron] * Pixels.Pixel24;
            value += Layer1Weights[Z, 24, Neuron] * Pixels.Pixel25;

            value += Layer1Biases[Z, Neuron];
            value = Hlsl.Tanh(value);
            value += Layer1OutputBiases[Z, Neuron];

            return value;
        }

        float4 GetLayer2Output(int Neuron, Layer1Data Layer1, int Z)
        {
            //Neuron = 9 * Pixel + Neuron;

            float4 value = default;
            value += Layer2Weights[Z, 0, Neuron] * Layer1.Neuron1;
            value += Layer2Weights[Z, 1, Neuron] * Layer1.Neuron2;
            value += Layer2Weights[Z, 2, Neuron] * Layer1.Neuron3;
            value += Layer2Weights[Z, 3, Neuron] * Layer1.Neuron4;
            value += Layer2Weights[Z, 4, Neuron] * Layer1.Neuron5;
            value += Layer2Weights[Z, 5, Neuron] * Layer1.Neuron6;
            value += Layer2Weights[Z, 6, Neuron] * Layer1.Neuron7;
            value += Layer2Weights[Z, 7, Neuron] * Layer1.Neuron8;
            value += Layer2Weights[Z, 8, Neuron] * Layer1.Neuron9;
            value += Layer2Weights[Z, 9, Neuron] * Layer1.Neuron10;
            value += Layer2Weights[Z, 10, Neuron] * Layer1.Neuron11;
            value += Layer2Weights[Z, 11, Neuron] * Layer1.Neuron12;
            value += Layer2Weights[Z, 12, Neuron] * Layer1.Neuron13;
            value += Layer2Weights[Z, 13, Neuron] * Layer1.Neuron14;
            value += Layer2Weights[Z, 14, Neuron] * Layer1.Neuron15;
            value += Layer2Weights[Z, 15, Neuron] * Layer1.Neuron16;
            value += Layer2Weights[Z, 16, Neuron] * Layer1.Neuron17;
            value += Layer2Weights[Z, 17, Neuron] * Layer1.Neuron18;

            value += Layer2Biases[Z, Neuron];
            value = Hlsl.Tanh(value);
            value += Layer2OutputBiases[Z, Neuron];

            return value;
        }

        //float4 GetLayer3Output(int Neuron, Layer2Data Layer2, int Z, int Pixel)
        //{
        //    Neuron = 3 * Pixel + Neuron;

        //    float4 value = default;
        //    value += Layer3Weights[Z, 0, Neuron] * Layer2.Neuron1;
        //    value += Layer3Weights[Z, 1, Neuron] * Layer2.Neuron2;
        //    value += Layer3Weights[Z, 2, Neuron] * Layer2.Neuron3;
        //    value += Layer3Weights[Z, 3, Neuron] * Layer2.Neuron4;
        //    value += Layer3Weights[Z, 4, Neuron] * Layer2.Neuron5;
        //    value += Layer3Weights[Z, 5, Neuron] * Layer2.Neuron6;
        //    value += Layer3Weights[Z, 6, Neuron] * Layer2.Neuron7;
        //    value += Layer3Weights[Z, 7, Neuron] * Layer2.Neuron8;
        //    value += Layer3Weights[Z, 8, Neuron] * Layer2.Neuron9;

        //    value += Layer3Biases[Z, Neuron];
        //    value = Hlsl.Tanh(value);
        //    value += Layer3OutputBiases[Z, Neuron];

        //    return value;
        //}

        float4 GetOutputPixel(int Neuron, Layer2Data Layer2, int Z)
        {
            float4 value = default;
            value += OutputLayerWeights[Z, 0, Neuron] * Layer2.Neuron1;
            value += OutputLayerWeights[Z, 1, Neuron] * Layer2.Neuron2;
            value += OutputLayerWeights[Z, 2, Neuron] * Layer2.Neuron3;

            value += OutputLayerBiases[Z, Neuron];

            return value;
        }
    }

    [ThreadGroupSize(DefaultThreadGroupSizes.X)]
    [GeneratedComputeShaderDescriptor]
    [RequiresDoublePrecisionSupport]
    public readonly partial struct SumErrors(ReadWriteTexture3D<float> CalculatedErrors, ReadWriteBuffer<double> ErrorTotals) : IComputeShader
    {
        public void Execute()
        {
            //double total = 0;

            //for (int x = 0; x < Width; x++)
            //    for (int y = 0;  y < Height; y++)
            //        total += CalculatedErrors[x, y, ThreadIds.X];

            ErrorTotals[ThreadIds.X] = CalculatedErrors[0,0,ThreadIds.X];
        }
    }

    public record OutputBitmaps (Bitmap Bitmap8, SKBitmap Bitmap16);

    [ThreadGroupSize(DefaultThreadGroupSizes.XY)]
    [GeneratedComputeShaderDescriptor]
    public readonly partial struct Reduce(
    ReadWriteTexture3D<float> InputTexture,
    ReadWriteTexture3D<float> OutputTexture,
    int InputWidth,
    int InputHeight) : IComputeShader
    {
        public void Execute()
        {
            // ThreadIds map to the OUTPUT texture coordinates
            int outX = ThreadIds.X;
            int outY = ThreadIds.Y;
            int z = ThreadIds.Z; // The model index

            // Calculate the top-left corner of the 2x2 block in the input
            int inX = outX * 2;
            int inY = outY * 2;

            float sum = 0;

            // Safely sample the 2x2 block, treating out-of-bounds as 0
            if (inX < InputWidth && inY < InputHeight)
                sum += InputTexture[inX, inY, z];

            if (inX + 1 < InputWidth && inY < InputHeight)
                sum += InputTexture[inX + 1, inY, z];

            if (inX < InputWidth && inY + 1 < InputHeight)
                sum += InputTexture[inX, inY + 1, z];

            if (inX + 1 < InputWidth && inY + 1 < InputHeight)
                sum += InputTexture[inX + 1, inY + 1, z];

            OutputTexture[outX, outY, z] = sum;
        }
    }
}
