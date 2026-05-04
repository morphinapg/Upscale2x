using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Rendering.Composition;
using ComputeSharp;
using SkiaSharp;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Tmds.DBus.Protocol;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace Upscale2x.ViewModels
{
    [DataContract]
    public class Evolution : ViewModelBase
    {
        public const int NumberOfModels = 30, NumberOfTrees = 2;

        /// <summary>
        /// All family trees, each with a number of models
        /// </summary>
        [DataMember]
        public List<NeuralNetwork>[] FamilyTrees = new List<NeuralNetwork>[NumberOfTrees];

        /// <summary>
        /// The top performing models of all time
        /// </summary>
        [DataMember]
        public NeuralNetwork[][] TopModels = new NeuralNetwork[NumberOfTrees][];

        public NeuralNetwork TopModel => TopModels[0][0];

        /// <summary>
        /// Whether the top model has changed this generation
        /// </summary>
        public bool TopModelChanged = true;

        [DataMember]
        DateTime? _lastUpdate = null;
        /// <summary>
        /// The last time the top model was updated
        /// </summary>
        public DateTime? LastUpdate
        {
            get => _lastUpdate;
            set
            {
                _lastUpdate = value;
                OnPropertyChanged(nameof(LastUpdate));
            }
        }



        

        [DataMember]
        public int Generations = 0;

        [DataMember]
        public bool RefineModel;


        /// <summary>
        /// The Accuracy of the Evolution model at the start of training
        /// </summary>
        public double? OldError;

        /// <summary>
        /// Text representing how long it's been since the model updated
        /// </summary>
        public string? LastUpdateText
        {
            get
            {
                if (LastUpdate is null)
                    return null;

                var Time = DateTime.Now - LastUpdate.Value;

                if (Time.TotalSeconds > 60 && TopModelLocked == false)
                {
                    TopModelLocked = true;
                    OnPropertyChanged(nameof(Checkmark));
                }


                if (Time.TotalSeconds < 2)
                    return "(Updated 1 second ago)";
                else if (Time.TotalSeconds < 60)
                    return "(Updated " + Time.Seconds + " seconds ago)";
                else if (Time.TotalMinutes < 2)
                    return "(Updated 1 minute ago)";
                else if (Time.TotalMinutes < 60)
                    return "(Updated " + Time.Minutes + " minutes ago)";
                else if (Time.TotalHours < 2)
                    return "(Updated 1 hour ago)";
                else if (Time.TotalHours < 24)
                    return "(Updated " + Time.Hours + " hours ago)";
                else if (Time.TotalDays < 2)
                    return "(Updated 1 day ago)";
                else
                    return "(Updated " + Time.Days + " days ago)";
            }
        }

        /// <summary>
        /// This represents whether a model has remained consistent for at least 60 seconds at any time during training, suggesting that the model is settling on a possible optimal state
        /// </summary>
        public bool? TopModelLocked = null;

        /// <summary>
        /// Displays a checkmark next to the network name on HomePage if TopModelLocked is ever set to true
        /// </summary>
        public string? Checkmark => TopModelLocked == true && OldError is not null ? "✔️" : null;

        /// <summary>
        /// Forces the UI to refresh the Last Update Text
        /// </summary>
        public void UpdateText()
        {
            OnPropertyChanged(nameof(LastUpdateText));
        }

        [DataMember]
        float[]
            Averages, Deviations, OutputBias;

        /// <summary>
        /// Initialize Evolution model
        /// </summary>
        /// <param name="network">Parent Network</param>
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider adding the 'required' modifier or declaring as nullable.
        public Evolution(bool refine)
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider adding the 'required' modifier or declaring as nullable.
        {
            RefineModel = refine;
            
            ResetEvolution();
        }

        

        public bool ResetSecondary = false;

        /// <summary>
        /// Whenever a model from the second branch is better than the worst model from the first, 
        /// add all better performing models and then trim to NumberOfModels
        /// </summary>
        public void Crossover()
        {
            ResetSecondary = false;
            var ModelToBeat = FamilyTrees[0][1];
            var ModelsToAdd = FamilyTrees[1].Where(x => x > ModelToBeat);
            FamilyTrees[0].AddRange(ModelsToAdd);
            FamilyTrees[0].Sort();
            if (FamilyTrees[0].Count > NumberOfModels)
            {
                var NumberToRemove = FamilyTrees[0].Count - NumberOfModels;
                FamilyTrees[0].RemoveRange(30, NumberToRemove);
                ResetSecondary = true;
                TopModels[1] = new NeuralNetwork[2];
            }
        }

        /// <summary>
        /// Check if the top two performing models have improved in either family tree
        /// </summary>
        /// <param name="i">Family Tree index</param>
        public void UpdateTopModel(int i)
        {
            if (FamilyTrees[i][0] > TopModels[i][0])
            {
                //Move to second place
                TopModels[i][1] = TopModels[i][0];

                //Update top model
                TopModels[i][0] = FamilyTrees[i][0];
                if (i == 0)
                {
                    TopModelChanged = true;
                    LastUpdate = DateTime.Now;

                    if (TopModelLocked is null)
                        TopModelLocked = false;
                }

                //check if second child beats new second place as well
                if (FamilyTrees[i][1] > TopModels[i][1])
                    TopModels[i][1] = FamilyTrees[i][1];
            }
            else if (FamilyTrees[i][0] > TopModels[i][1])
                TopModels[i][1] = FamilyTrees[i][1];
        }

        const int NumberOfParents = 32;

        /// <summary>
        /// Retrieve a selection of 4 parent candidates for breeding
        /// The first two are the top two performing models of all time
        /// The second two are the top two performing models
        /// </summary>
        public List<NeuralNetwork>[] GetParents()
        {
            var Parents = new List<NeuralNetwork>[NumberOfTrees];
            for (int i = 0; i < NumberOfTrees; i++)
            {
                Parents[i] = new();

                //Skip choosing parents on the secondary branch if ResetSecondary is true
                if (i == 0 || !ResetSecondary)
                {
                    if (TopModels[i][0] is not null)
                        Parents[i].Add(TopModels[i][0]);
                    if (TopModels[i][1] is not null)
                        Parents[i].Add(TopModels[i][1]);

                    for (int j = 0; j < NumberOfModels && Parents[i].Count < NumberOfParents; j++)
                        if (FamilyTrees[i][j] < Parents[i].Last())
                            Parents[i].Add(FamilyTrees[i][j]);

                    //In the rare case where there are not two unique models from FamilyTrees in addition to the top two models
                    while (Parents[i].Count < NumberOfParents)
                        Parents[i].Add(new NeuralNetwork(Averages, Deviations, OutputBias));
                }
            }

            return Parents;
        }

        /// <summary>
        /// Choose two parents from 4 possibilities, and combine the two to produce a child
        /// </summary>
        /// <param name="x">Family Tree index</param>
        /// <param name="y">Model Index</param>
        /// <param name="Parents">The 4 parents to choose from</param>
        public void Breed(int x, int y, List<NeuralNetwork>[] Parents)
        {
            if (x == 1 && ResetSecondary)
            {
                //Reset the secondary branch if models have crossed over to the first branch
                FamilyTrees[x][y] = new NeuralNetwork(Averages, Deviations, OutputBias);
            }
            else
            {
                //Create a new child by breeding two parents
                var r = Random.Shared;

                //Choose a random number of parents to select from
                int SelectionGroup = r.Next(NumberOfParents) + 1;

                if (SelectionGroup < 4)
                    SelectionGroup = 4;

                //Choose two parents from the selection group
                int
                    Parent1 = r.Next(SelectionGroup),
                    Parent2 = r.Next(SelectionGroup);

                //if (Parent2 >= Parent1)
                //    Parent2++;

                if (Parent1 == Parent2)
                    FamilyTrees[x][y] = new NeuralNetwork(Parents[x][Parent1]);
                else
                    FamilyTrees[x][y] = Parents[x][Parent1] + Parents[x][Parent2];
            }
        }

        public double? Error => TopModel is not null ? TopModel.Error : null;               //Representation of how many incorrect predictions there were, and by how much
        
        /// <summary>
        /// Deep clone other Evolution model
        /// </summary>
        public Evolution(Evolution other)
        {
            Averages = other.Averages;
            Deviations = other.Deviations;
            OutputBias = other.OutputBias;

            LastUpdate = other.LastUpdate;
            RefineModel = other.RefineModel;

            Generations = other.Generations;

            for (int i = 0; i < NumberOfTrees; i++)
            {
                FamilyTrees[i] = new();

                foreach (var model in other.FamilyTrees[i].ToList())
                {
                    FamilyTrees[i].Add(model);
                }

                TopModels[i] = new NeuralNetwork[2];
                TopModels[i][0] = other.TopModels[i][0];
                TopModels[i][1] = other.TopModels[i][1];
            }
        }

        public void NextGeneration(ReadWriteTexture2D<Rgba64, float4>? InputImage, ReadWriteTexture2D<Rgba64, float4>? DownscaledImage, bool RefineMode, NeuralNetwork? BaseModel = null)
        {
            Generations++;

            // STEP 1 - ACCURACY TESTING //
            //First, we need to test every NeuralNetwork in every Evolution model for accuracy
            //This will leverage the GPU
            TestAccuracy(InputImage, DownscaledImage, RefineMode, BaseModel);

            // STEP 2 - SORTING //
            Parallel.ForEach(FamilyTrees, x => x.Sort());

            // STEP 3 - CROSSOVER //
            Crossover();

            // STEP 4 - UPDATE TOP MODELS //
            Parallel.For(0, NumberOfTrees, i => UpdateTopModel(i));

            // STEP 5 - EXTRACT PARAMETERS & CALCULATE TRENDS
            int paramCount = FamilyTrees[0][0].GetFlatParameterCount();
            float[][] inputParams = new float[NumberOfTrees][];
            float[][] targetParams = new float[NumberOfTrees][];
            double[][] treeErrors = new double[NumberOfTrees][];
            double[] sumX = new double[NumberOfTrees];
            double[] denominator = new double[NumberOfTrees];

            Parallel.For(0, NumberOfTrees, i =>
            {
                inputParams[i] = new float[paramCount * NumberOfModels];
                targetParams[i] = new float[paramCount];
                treeErrors[i] = new double[NumberOfModels];

                NeuralNetwork.ExtractTreeParameters(FamilyTrees[i], inputParams[i], treeErrors[i], out sumX[i], out denominator[i]);
                NeuralNetwork.CalculateTrendParameters(inputParams[i], targetParams[i], treeErrors[i], paramCount, NumberOfModels, sumX[i], denominator[i]);
            });

            // STEP 6 - RETRIEVE PARENTS
            var Parents = GetParents();

            // STEP 5 - BREEDING //
            //var Parents = GetParents();
            // Parallel.ForEach(Enumerable.Range(0, NumberOfTrees).Select(x => Enumerable.Range(0, NumberOfModels).Select(y => (new {x, y}))).SelectMany(x => x), model => Breed(model.x, model.y, Parents));

            // STEP 7 - BREEDING & PROJECTION
            Parallel.For(0, NumberOfTrees * NumberOfModels, flatIndex =>
            {
                int treeIndex = flatIndex / NumberOfModels;
                int modelIndex = flatIndex % NumberOfModels;

                // NEW: Safely handle the crossover reset for the secondary branch
                if (treeIndex == 1 && ResetSecondary)
                {
                    FamilyTrees[treeIndex][modelIndex] = new NeuralNetwork(Averages, Deviations, OutputBias);
                }
                else if (modelIndex < 4)
                {
                    // Inject our mathematically perfect targets into the first 4 slots
                    FamilyTrees[treeIndex][modelIndex] = Parents[treeIndex][modelIndex].ProjectModel(targetParams[treeIndex]);
                }
                else
                {
                    // Standard breeding for the remainder of the generation
                    Breed(treeIndex, modelIndex, Parents);
                }
            });

            // STEP 8 - MUTATION //
            MutateModels();
        }

        void MutateModels()
        {
            var r = Random.Shared;

            if (!TopModelChanged)
            {
                //if the top model has not changed, we either need to increase the mutation rate or intensity of one of the models to introduce variety
                var ModelToModify = FamilyTrees[0][r.Next(4, NumberOfModels)];

                if (r.NextSingle() < 0.5f)
                {
                    //Increase mutation rate
                    ModelToModify.IncreaseMutationRate();
                }
                else
                {
                    //Increase mutation intensity
                    ModelToModify.IncreaseMutationIntensity();
                }
            }

            // Mutate only the standard bred models (skip indices 0 through 3)
            var ModelsToMutate = FamilyTrees.SelectMany(x => x.Skip(4)).Where(x => x is not null);
            Parallel.ForEach(ModelsToMutate, x => x.MutateModel());

            //var AllModels = FamilyTrees.SelectMany(x => x);
            //Parallel.ForEach(FamilyTrees.SelectMany(x => x), x => x.MutateModel());
            //Parallel.ForEach(FamilyTrees.Where(x => !x.Where(y => y.IsMutated).Any()), x => x[r.Next(NumberOfModels)].IncreaseMutationRate());            

            // Fallback if no models mutated
            Parallel.For(0, NumberOfTrees, i =>
            {
                var tree = FamilyTrees[i];
                if (!tree.Any(y => y.IsMutated))
                {
                    tree[r.Next(4, NumberOfModels)].IncreaseMutationRate();
                }
            });
        }

        

        public void ResetEvolution()
        {
            Averages = Enumerable.Repeat(0f, 25).ToArray();
            Deviations = Enumerable.Repeat(1f, 25).ToArray();
            OutputBias = Enumerable.Repeat(0f, 4).ToArray();
            Parallel.For(0, NumberOfTrees, i =>
            {
                FamilyTrees[i] = new();
                for (int j = 0; j < NumberOfModels; j++)
                    FamilyTrees[i].Add(new NeuralNetwork(Averages, Deviations, OutputBias));
            });

            Parallel.For(0, 2, i => TopModels[i] = new NeuralNetwork[2]);

            OldError = null;
            Generations = 0;

            MutateModels();
        }
        


        public async Task<OutputBitmaps?> Upscale(ReadWriteTexture2D<Rgba64, float4>? InputImage, bool AcceleratedMode, bool AnalyzeRender, NeuralNetwork? BaseModel = null)
        {
            if (InputImage is not null)
            {
                var Model = TopModel is not null ? TopModel : FamilyTrees.SelectMany(x => x).Where(x => x is not null).FirstOrDefault();

                if ( Model is not null )
                {
                    var Stats = Model.GetModelStats();

                    var outputs = await Task.Run(() =>
                    {
                        var GPU = GraphicsDevice.GetDefault();
                        int
                            NewHeight = InputImage.Height * 2,
                            NewWidth = InputImage.Width * 2;

                        using var OutputImage = GPU.AllocateReadWriteTexture2D<float4>(NewWidth, NewHeight);

                        var Params = Model.GetModelParameters();

                        var UseBase = BaseModel is not null;
                        var BaseParams = BaseModel is not null ? BaseModel.GetModelParameters() : Params;
                        var models = UseBase ? 2 : 1;

                        float[,]
                                InputAverages_CPU = new float[Stats.Layer1Inputs * models, 1],
                                InputDeviations_CPU = new float[Stats.Layer1Inputs * models, 1],
                                Layer1Biases_CPU = new float[Stats.Layer1Size * models, 1],
                                Layer1OutputBiases_CPU = new float[Stats.Layer1Size * models, 1],
                                Layer2Biases_CPU = new float[Stats.Layer2Size * models, 1],
                                Layer2OutputBiases_CPU = new float[Stats.Layer2Size * models, 1],
                                OutputLayerBiases_CPU = new float[Stats.Outputs * models, 1];

                        float[,,]
                            Layer1Weights_CPU = new float[Stats.Layer1Size * models, Stats.Layer1Inputs, 1],
                            Layer2Weights_CPU = new float[Stats.Layer2Size * models, Stats.Layer2Inputs, 1],
                            OutputLayerWeights_CPU = new float[Stats.Outputs * models, Stats.OutputInputs, 1];

                        //Input Layer
                        for (int x = 0; x < Stats.Layer1Inputs; x++)
                        {
                            InputAverages_CPU[x, 0] = Params.InputAverages[x];
                            InputDeviations_CPU[x, 0] = Params.InputDeviations[x];

                            if (UseBase)
                            {
                                InputAverages_CPU[x + Stats.Layer1Inputs, 0] = BaseParams.InputAverages[x];
                                InputDeviations_CPU[x + Stats.Layer1Inputs, 0] = BaseParams.InputDeviations[x];
                            }
                        }

                        //Layer 1
                        for (int x = 0; x < Stats.Layer1Size; x++)
                        {
                            for (int y = 0; y < Stats.Layer1Inputs; y++)
                                Layer1Weights_CPU[x, y, 0] = Params.Layer1Weights[x, y];

                            Layer1Biases_CPU[x, 0] = Params.Layer1Biases[x];
                            Layer1OutputBiases_CPU[x, 0] = Params.Layer1OutputBiases[x];

                            if (UseBase)
                            {
                                for (int y = 0; y < Stats.Layer1Inputs; y++)
                                    Layer1Weights_CPU[x + Stats.Layer1Size, y, 0] = BaseParams.Layer1Weights[x, y];
                                Layer1Biases_CPU[x + Stats.Layer1Size, 0] = BaseParams.Layer1Biases[x];
                                Layer1OutputBiases_CPU[x + Stats.Layer1Size, 0] = BaseParams.Layer1OutputBiases[x];
                            }
                        }

                        //Layer 2
                        for (int x = 0; x < Stats.Layer2Size; x++)
                        {
                            for (int y = 0; y < Stats.Layer2Inputs; y++)
                                Layer2Weights_CPU[x, y, 0] = Params.Layer2Weights[x, y];

                            Layer2Biases_CPU[x, 0] = Params.Layer2Biases[x];
                            Layer2OutputBiases_CPU[x, 0] = Params.Layer2OutputBiases[x];

                            if (UseBase)
                            {
                                for (int y = 0; y < Stats.Layer2Inputs; y++)
                                    Layer2Weights_CPU[x + Stats.Layer2Size, y, 0] = BaseParams.Layer2Weights[x, y];
                                Layer2Biases_CPU[x + Stats.Layer2Size, 0] = BaseParams.Layer2Biases[x];
                                Layer2OutputBiases_CPU[x + Stats.Layer2Size, 0] = BaseParams.Layer2OutputBiases[x];
                            }
                        }

                        //Output Layer
                        for (int x = 0; x < Stats.Outputs; x++)
                        {
                            for (int y = 0; y < Stats.OutputInputs; y++)
                                OutputLayerWeights_CPU[x, y, 0] = Params.OutputLayerWeights[x, y];

                            OutputLayerBiases_CPU[x, 0] = Params.OutputLayerBiases[x];

                            if (UseBase)
                            {
                                for (int y = 0; y < Stats.OutputInputs; y++)
                                    OutputLayerWeights_CPU[x + Stats.Outputs, y, 0] = BaseParams.OutputLayerWeights[x, y];
                                OutputLayerBiases_CPU[x + Stats.Outputs, 0] = BaseParams.OutputLayerBiases[x];
                            }
                        }

                        using var InputAverages = GPU.AllocateReadOnlyTexture2D(InputAverages_CPU);
                        using var InputDeviations = GPU.AllocateReadOnlyTexture2D(InputDeviations_CPU);
                        using var Layer1Weights = GPU.AllocateReadOnlyTexture3D(Layer1Weights_CPU);
                        using var Layer1Biases = GPU.AllocateReadOnlyTexture2D(Layer1Biases_CPU);
                        using var Layer1OutputBiases = GPU.AllocateReadOnlyTexture2D(Layer1OutputBiases_CPU);
                        using var Layer2Weights = GPU.AllocateReadOnlyTexture3D(Layer2Weights_CPU);
                        using var Layer2Biases = GPU.AllocateReadOnlyTexture2D(Layer2Biases_CPU);
                        using var Layer2OutputBiases = GPU.AllocateReadOnlyTexture2D(Layer2OutputBiases_CPU);
                        using var OutputLayerWeights = GPU.AllocateReadOnlyTexture3D(OutputLayerWeights_CPU);
                        using var OutputLayerBiases = GPU.AllocateReadOnlyTexture2D(OutputLayerBiases_CPU);

                        using var dummyoutput = GPU.AllocateReadWriteTexture3D<float>(2, 2, 1);



                        GPU.For(InputImage.Width, InputImage.Height, new Upscale_GPU(
                            InputImage,
                            InputImage,
                            dummyoutput,
                            OutputImage,
                            false,
                            true,
                            AcceleratedMode,
                            AnalyzeRender,
                            UseBase,
                            InputAverages,
                            InputDeviations,
                            Layer1Weights,
                            Layer1Biases,
                            Layer1OutputBiases,
                            Layer2Weights,
                            Layer2Biases,
                            Layer2OutputBiases,
                            OutputLayerWeights,
                            OutputLayerBiases
                            ));

                        var bitmap8 = GetBitmap8(OutputImage);

                        var bitmap16 = GetBitmap16(OutputImage);

                        return new OutputBitmaps(bitmap8, bitmap16);
                    });

                    return outputs;
                }

                return null;
            }

            return null;
        }

        public Bitmap GetBitmap8(ReadWriteTexture2D<Float4> OutputImage)
        {
            int
                NewWidth = OutputImage.Width,
                NewHeight = OutputImage.Height;

            var cpuData = new float4[NewWidth * NewHeight];
            OutputImage.CopyTo(cpuData);

            var bitmapData = new byte[NewWidth * NewHeight * 4]; // 4 bytes per pixel (RGBA)

            for (int i = 0; i < cpuData.Length; i++)
            {
                float4 pixel = cpuData[i];

                // Clamp and convert float to byte (0-255)
                byte r = (byte)(Math.Clamp(pixel.R, 0.0f, 1.0f) * 255);
                byte g = (byte)(Math.Clamp(pixel.G, 0.0f, 1.0f) * 255);
                byte b = (byte)(Math.Clamp(pixel.B, 0.0f, 1.0f) * 255);
                byte a = (byte)(Math.Clamp(pixel.A, 0.0f, 1.0f) * 255);

                bitmapData[i * 4] = r;
                bitmapData[i * 4 + 1] = g;
                bitmapData[i * 4 + 2] = b;
                bitmapData[i * 4 + 3] = a;
            }

            IntPtr unmanagedPointer = Marshal.AllocHGlobal(bitmapData.Length);
            Marshal.Copy(bitmapData, 0, unmanagedPointer, bitmapData.Length);

            Bitmap bitmap = new Bitmap(PixelFormat.Rgba8888, AlphaFormat.Premul, unmanagedPointer, new Avalonia.PixelSize(NewWidth, NewHeight), new Avalonia.Vector(96, 96), NewWidth * 4);

            return bitmap;
        }

        public SKBitmap GetBitmap16(ReadWriteTexture2D<Float4> OutputImage)
        {
            int
                NewWidth = OutputImage.Width,
                NewHeight = OutputImage.Height;

            var _imageData16bpc = new Half[NewWidth * NewHeight * 4];

            var cpuData = new float4[NewWidth * NewHeight];
            OutputImage.CopyTo(cpuData);

            for (int i = 0; i < cpuData.Length; i++)
            {
                float4 pixel = cpuData[i];

                // Convert from float (0-1) to 16-bit unsigned integer (0-65535)
                _imageData16bpc[i * 4] = (Half)Math.Clamp(pixel.R, 0.0f, 1.0f);
                _imageData16bpc[i * 4 + 1] = (Half)Math.Clamp(pixel.G, 0.0f, 1.0f);
                _imageData16bpc[i * 4 + 2] = (Half)Math.Clamp(pixel.B, 0.0f, 1.0f);
                _imageData16bpc[i * 4 + 3] = (Half)Math.Clamp(pixel.A, 0.0f, 1.0f);
            }

            var bitmap = new SKBitmap(NewWidth, NewHeight, SKColorType.RgbaF16, SKAlphaType.Premul);

            IntPtr pixelPtr = bitmap.GetPixels();

            unsafe
            {
                fixed (Half* srcPtr = _imageData16bpc)
                {
                    Half* destPtr = (Half*)pixelPtr;
                    for (int i = 0; i < _imageData16bpc.Length; i++)
                    {
                        destPtr[i] = srcPtr[i];
                    }
                }
            }

            return bitmap;
        }

        public double? InitializeError(ReadWriteTexture2D<Rgba64, float4>? InputImage, ReadWriteTexture2D<Rgba64, float4>? DownscaledImage, bool RefineMode, NeuralNetwork? BaseModel = null)
        {
            TestAccuracy(InputImage, DownscaledImage, RefineMode, BaseModel, true);

            var Model = FamilyTrees.SelectMany(x => x).Where(x => x is not null && x.Error is not null).FirstOrDefault();


            return Model?.Error;
        }
        

        void TestAccuracy(ReadWriteTexture2D<Rgba64, float4>? InputImage, ReadWriteTexture2D<Rgba64, float4>? DownscaledImage, bool RefineMode, NeuralNetwork? BaseModel = null, bool Initialize = false)
        {
            var Tops = TopModels.SelectMany(x => x);
            var AllModels = FamilyTrees.SelectMany(x => x).Concat(Tops).Where(x => x is not null).ToList();

            if (Initialize) 
                AllModels = AllModels.Take(1).ToList();

            if (InputImage is not null && DownscaledImage is not null && AllModels.Any())
            {
                int UpscaledPixels = DownscaledImage.Width * 2 * DownscaledImage.Height * 2;
                            

                var Stats = AllModels.First().GetModelStats();

                float ModelSize = Stats.MemorySize + UpscaledPixels * 20f;

                var GPU = GraphicsDevice.GetDefault();
                var VRAM = (float)GPU.DedicatedMemorySize;

                long combinedLuid = long.Parse(GPU.Luid.ToString());
                long highPart = combinedLuid >> 32;
                long lowPart = combinedLuid & 0xFFFFFFFF; // Mask to get low 32 bits

                string instanceName = $"luid_0x{highPart:X8}_0x{lowPart:X8}_phys_0";

                var PC = new PerformanceCounter("GPU Adapter Memory", "Dedicated Usage", instanceName);

                var used = PC.NextValue();

                var AvailableVRAM = (VRAM - used) * 0.9f; //leave 10% overhead to be safe

                var ModelCount = NumberOfModels * NumberOfTrees;

                //The number of minimum bytes needed per training session
                var BaseBytes =
                    (InputImage.Width * InputImage.Height + DownscaledImage.Width * DownscaledImage.Height) * 16f           //Loading Original and downscaled image into memory
                    + (InputImage.Width * InputImage.Height * 80f)                                                          //Leave enough memory free to render while training
                    + 64f                                                                                                   //bytes reserved for a dummy texture
                    + ModelCount;                                                                                           //enough bytes for storing the calculated error totals

                AvailableVRAM -= BaseBytes;

                

                var BatchSize = (int)(AvailableVRAM / ModelSize);
                if (BatchSize > ModelCount)
                    BatchSize = ModelCount;

                //Test to see if the 3D texture would be too large
                bool Use2DTexture = false;

                int
                        NewWidth = DownscaledImage.Width * 2,
                        NewHeight = DownscaledImage.Height * 2;

                try
                {
                    using var test = GPU.AllocateReadWriteTexture3D<float4>(NewWidth, NewHeight, BatchSize);
                }
                catch
                {
                    Use2DTexture = true;
                }

                if (BatchSize == 0 || Use2DTexture)
                    BatchSize = 1;

                

                

                //BatchSize = 1;

                var TotalBatches = (int)(Math.Ceiling(ModelCount / (double)BatchSize));

                for (int i = 0; i < TotalBatches; i++)
                {
                    var NumberToSkip = i * BatchSize;
                    var CurrentBatch = AllModels.Skip(NumberToSkip).Take(BatchSize).Select(x =>
                    {
                        var Params = x.GetModelParameters();
                        return new { Model = x, Params };
                    }).ToList();

                    var CurrentBatchSize = CurrentBatch.Count;

                    var Current2DTexture = Use2DTexture || CurrentBatchSize == 1;

                    using var CalculatedErrors = Current2DTexture ? GPU.AllocateReadWriteTexture3D<float>(2, 2, 1) : GPU.AllocateReadWriteTexture3D<float>(NewWidth, NewHeight, CurrentBatchSize);
                    using var OutputTexture = Current2DTexture ? GPU.AllocateReadWriteTexture2D<float4>(NewWidth, NewHeight) : GPU.AllocateReadWriteTexture2D<float4>(2, 2);

                    int models = BaseModel is null ? 1 : 2;

                    float[,]
                        InputAverages_CPU = new float[Stats.Layer1Inputs * models, CurrentBatchSize],
                        InputDeviations_CPU = new float[Stats.Layer1Inputs * models, CurrentBatchSize],
                        Layer1Biases_CPU = new float[Stats.Layer1Size * models, CurrentBatchSize],
                        Layer1OutputBiases_CPU = new float[Stats.Layer1Size * models, CurrentBatchSize],
                        Layer2Biases_CPU = new float[Stats.Layer2Size * models, CurrentBatchSize],
                        Layer2OutputBiases_CPU = new float[Stats.Layer2Size * models, CurrentBatchSize],
                        OutputLayerBiases_CPU = new float[Stats.Outputs * models, CurrentBatchSize];

                    float[,,]
                        Layer1Weights_CPU = new float[Stats.Layer1Size * models, Stats.Layer1Inputs, CurrentBatchSize],
                        Layer2Weights_CPU = new float[Stats.Layer2Size * models, Stats.Layer2Inputs, CurrentBatchSize],
                        OutputLayerWeights_CPU = new float[Stats.Outputs * models, Stats.OutputInputs, CurrentBatchSize];

                    var UseBase = BaseModel is not null;
                    var BaseParams = BaseModel is not null ? BaseModel.GetModelParameters() : CurrentBatch[0].Params;

                    Parallel.For(0, CurrentBatchSize, z =>
                    {
                        //Input Layer
                        for (int x = 0; x < Stats.Layer1Inputs; x++)
                        {
                            InputAverages_CPU[x, z] = CurrentBatch[z].Params.InputAverages[x];
                            InputDeviations_CPU[x, z] = CurrentBatch[z].Params.InputDeviations[x];

                            if (UseBase)
                            {
                                InputAverages_CPU[x + Stats.Layer1Inputs, z] = BaseParams.InputAverages[x];
                                InputDeviations_CPU[x + Stats.Layer1Inputs, z] = BaseParams.InputDeviations[x];
                            }
                        }

                        //Layer 1
                        for (int x = 0; x < Stats.Layer1Size; x++)
                        {
                            for (int y = 0; y < Stats.Layer1Inputs; y++)
                                Layer1Weights_CPU[x, y, z] = CurrentBatch[z].Params.Layer1Weights[x, y];

                            Layer1Biases_CPU[x, z] = CurrentBatch[z].Params.Layer1Biases[x];
                            Layer1OutputBiases_CPU[x,z] = CurrentBatch[z].Params.Layer1OutputBiases[x];

                            if (UseBase)
                            {
                                for (int y = 0; y < Stats.Layer1Inputs; y++)
                                    Layer1Weights_CPU[x + Stats.Layer1Size, y, z] = BaseParams.Layer1Weights[x, y];
                                Layer1Biases_CPU[x + Stats.Layer1Size, z] = BaseParams.Layer1Biases[x];
                                Layer1OutputBiases_CPU[x + Stats.Layer1Size, z] = BaseParams.Layer1OutputBiases[x];
                            }
                        }

                        //Layer 2
                        for (int x = 0; x < Stats.Layer2Size; x++)
                        {
                            for (int y = 0; y < Stats.Layer2Inputs; y++)
                                Layer2Weights_CPU[x, y, z] = CurrentBatch[z].Params.Layer2Weights[x, y];

                            Layer2Biases_CPU[x, z] = CurrentBatch[z].Params.Layer2Biases[x];
                            Layer2OutputBiases_CPU[x, z] = CurrentBatch[z].Params.Layer2OutputBiases[x];

                            if (UseBase)
                            {
                                for (int y = 0; y < Stats.Layer2Inputs; y++)
                                    Layer2Weights_CPU[x + Stats.Layer2Size, y, z] = BaseParams.Layer2Weights[x, y];
                                Layer2Biases_CPU[x + Stats.Layer2Size, z] = BaseParams.Layer2Biases[x];
                                Layer2OutputBiases_CPU[x + Stats.Layer2Size, z] = BaseParams.Layer2OutputBiases[x];
                            }
                        }

                        //Output Layer
                        for (int x = 0; x < Stats.Outputs; x++)
                        {
                            for (int y = 0; y < Stats.OutputInputs; y++)
                                OutputLayerWeights_CPU[x, y, z] = CurrentBatch[z].Params.OutputLayerWeights[x, y];

                            OutputLayerBiases_CPU[x, z] = CurrentBatch[z].Params.OutputLayerBiases[x];

                            if (UseBase)
                            {
                                for (int y = 0; y < Stats.OutputInputs; y++)
                                    OutputLayerWeights_CPU[x + Stats.Outputs, y, z] = BaseParams.OutputLayerWeights[x, y];
                                OutputLayerBiases_CPU[x + Stats.Outputs, z] = BaseParams.OutputLayerBiases[x];
                            }
                        }
                    });

                    using var InputAverages = GPU.AllocateReadOnlyTexture2D(InputAverages_CPU);
                    using var InputDeviations = GPU.AllocateReadOnlyTexture2D(InputDeviations_CPU);
                    using var Layer1Weights = GPU.AllocateReadOnlyTexture3D(Layer1Weights_CPU);
                    using var Layer1Biases = GPU.AllocateReadOnlyTexture2D(Layer1Biases_CPU);
                    using var Layer1OutputBiases = GPU.AllocateReadOnlyTexture2D(Layer1OutputBiases_CPU);
                    using var Layer2Weights = GPU.AllocateReadOnlyTexture3D(Layer2Weights_CPU);
                    using var Layer2Biases = GPU.AllocateReadOnlyTexture2D(Layer2Biases_CPU);
                    using var Layer2OutputBiases = GPU.AllocateReadOnlyTexture2D(Layer2OutputBiases_CPU);
                    using var OutputLayerWeights = GPU.AllocateReadOnlyTexture3D(OutputLayerWeights_CPU);
                    using var OutputLayerBiases = GPU.AllocateReadOnlyTexture2D(OutputLayerBiases_CPU);

                    GPU.For(DownscaledImage.Width, DownscaledImage.Height, CurrentBatchSize, new Upscale_GPU(
                        DownscaledImage,
                        InputImage,
                        CalculatedErrors,
                        OutputTexture,
                        true,                        
                        Current2DTexture,
                        RefineMode,
                        false,
                        UseBase,
                        InputAverages,
                        InputDeviations,
                        Layer1Weights,
                        Layer1Biases,
                        Layer1OutputBiases,
                        Layer2Weights,
                        Layer2Biases,
                        Layer2OutputBiases,
                        OutputLayerWeights,
                        OutputLayerBiases
                        ));

                    //if (Current2DTexture)
                    //{
                    //    var AllErrors = new float4[UpscaledPixels];

                    //    OutputTexture.CopyTo(AllErrors);

                    //    var ErrorTotal = AllErrors.AsParallel().Select(x => x.R).Sum();

                    //    CurrentBatch.First().Model.Error = ErrorTotal;
                    //}
                    //else
                    //{
                    //    using var ErrorTotals = GPU.AllocateReadWriteBuffer<double>(CurrentBatchSize);

                    //    GPU.For(CurrentBatchSize, new SumErrors(CalculatedErrors, ErrorTotals, NewWidth, NewHeight));

                    //    var CalculatedTotals = new double[CurrentBatchSize];
                    //    ErrorTotals.CopyTo(CalculatedTotals);

                    //    Parallel.For(0, CurrentBatchSize, model => CurrentBatch[model].Model.Error = CalculatedTotals[model]);
                    //}
                    //

                    //Use Reduce Shader to sum the errors directly on the GPU

                    //We need a texture which will be used to swap with the CalculatedErrors texture during the reduction process.
                    //The dimensions of this texture will be halved each iteration, so we round up to handle odd dimensions
                    int
                        CurrentW = NewWidth , 
                        CurrentH = NewHeight;
                    

                    if (Current2DTexture)
                    {
                        using var ReductionTexture = GPU.AllocateReadWriteTexture2D<float>(CurrentW, CurrentH);

                        //If we're using a 2D texture, we need to process the errors in 2D
                        //first, convert the float4 error texture to a float texture (R channel contains the error value and the GBA channels are ignored)

                        using var OutputErrors = GPU.AllocateReadWriteTexture2D<float>(NewWidth, NewHeight);
                        GPU.For(NewWidth, NewHeight, new ConvertErrors(OutputTexture, OutputErrors));

                        //We will use references to swap between the two
                        var SwapInput = OutputErrors;
                        var SwapOutput = ReductionTexture;

                        while (CurrentW > 1 || CurrentH > 1)
                        {
                            // Round up to handle odd resolutions
                            int nextW = (CurrentW + 1) / 2;
                            int nextH = (CurrentH + 1) / 2;
                            GPU.For(nextW, nextH, new Reduce2D(SwapInput, SwapOutput, CurrentW, CurrentH));

                            // Ping-Pong the references! Zero copy overhead.
                            var temp = SwapInput;
                            SwapInput = SwapOutput;
                            SwapOutput = temp;

                            CurrentW = nextW;
                            CurrentH = nextH;
                        }

                        using var ErrorTotal = GPU.AllocateReadWriteBuffer<double>(1);
                        GPU.For(1, new SumErrors2D(SwapInput, ErrorTotal));

                        // Copy the calculated error totals back to the CPU and assign them to the models
                        var CalculatedTotals = new double[CurrentBatchSize];
                        ErrorTotal.CopyTo(CalculatedTotals);

                        CurrentBatch[0].Model.Error = CalculatedTotals[0];
                    }
                    else
                    {
                        using var ReductionTexture = GPU.AllocateReadWriteTexture3D<float>(CurrentW, CurrentH, CurrentBatchSize);

                        //We will use references to swap between the two
                        var SwapInput = CalculatedErrors;
                        var SwapOutput = ReductionTexture;

                        while (CurrentW > 1 || CurrentH > 1)
                        {
                            // Round up to handle odd resolutions
                            int nextW = (CurrentW + 1) / 2;
                            int nextH = (CurrentH + 1) / 2;
                            GPU.For(nextW, nextH, CurrentBatchSize, new Reduce(SwapInput, SwapOutput, CurrentW, CurrentH));

                            // Ping-Pong the references! Zero copy overhead.
                            var temp = SwapInput;
                            SwapInput = SwapOutput;
                            SwapOutput = temp;

                            CurrentW = nextW;
                            CurrentH = nextH;
                        }

                        // Use SumErrors shader to sum the final values for each model into a single error total
                        using var ErrorTotals = GPU.AllocateReadWriteBuffer<double>(CurrentBatchSize);
                        GPU.For(CurrentBatchSize, new SumErrors(SwapInput, ErrorTotals));

                        // Copy the calculated error totals back to the CPU and assign them to the models
                        var CalculatedTotals = new double[CurrentBatchSize];
                        ErrorTotals.CopyTo(CalculatedTotals);

                        Parallel.For(0, CurrentBatchSize, model => CurrentBatch[model].Model.Error = CalculatedTotals[model]);
                    }
                    
                }
            }
        }

        public void ResetErrors()
        {
            var Tops = TopModels.SelectMany(x => x);

            var AllModels = FamilyTrees.SelectMany(x => x).Concat(Tops).Where(x => x is not null).ToList();

            Parallel.ForEach(AllModels, x => x.Error = null);

            OldError = null;
        }
    }

    [DataContract]
    public class ModelBundle
    {
        [DataMember]
        public Evolution? BaseModel, RefineModel;

        [DataMember]
        public bool RefineMode;
    }
}
