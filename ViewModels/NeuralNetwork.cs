using Avalonia.Animation;
using ComputeSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace Upscale2x.ViewModels
{
    [DataContract]
    public class NeuralNetwork : Breedable, IComparable<NeuralNetwork>
    {
        [DataMember]
        public float[] InputAverage, InputDeviation, OutputBias;                                                        //Values used to pre-normalize the input values, and adjust output values

        [DataMember]
        int InputCount, OutputCount;                                                                                    //How many inputs and outputs the neural network will be capable of handling

        [DataMember]
        List<int> LayerSizes;                                                                                           //Defined number of neurons in each hidden layer

        [DataMember]
        List<Neuron[]> HiddenLayers;                                                                                    //A list including every hidden layer

        [DataMember]
        Neuron[] OutputLayer;                                                                                           //Array of neurons used for the output(s)

        [DataMember]
        float mutationrate, mutationintensity, TrendConfidence;                                                                         //values for controlling evolution

        [DataMember]
        public bool IsMutated = false;

        /// <summary>
        /// Creates a new Neural Network
        /// </summary>
        /// <param name="inputs">Number of Inputs</param>
        /// <param name="average">average value for each input, with a list for each input type</param>
        /// <param name="deviation">deviation value for each input, with a list for each input type</param>
        /// <param name="outputbias">average value of what the outputs should be, with a list for each input type</param>
        public NeuralNetwork(float[] average, float[] deviation, float[] outputbias)
        {
            if (average.Length == 0 || deviation.Length == 0)
                throw new Exception("Average and/or Deviation values missing!");
            if (average.Length != deviation.Length)
                throw new Exception("Input Types mismatch on Average and Deviation!");
            //if (average.Concat(deviation).Where(x => x.Length != average[0].Length).Any())
            //    throw new Exception("Number of inputs does not match on every instance of average/deviation!");
            //if (average.Count != outputbias.Count)
            //    throw new Exception("Input and output types do not match!");
            //if (outputbias.Where(x => x.Length != outputbias[0].Length).Any())
            //    throw new Exception("Number of outputs does not match for every outputbias!");

            InputCount = average.Length;
            OutputCount = outputbias.Length;

            InputAverage = average.ToArray();
            InputDeviation = deviation.ToArray();
            OutputBias = outputbias.ToArray();

            int CurrentLayer = (int)Math.Round(InputCount * 2.0 / 3 + OutputCount, 0);
            //int CurrentLayer = (int)Math.Round(InputCount * 2.0 / 3 + 1, 0);

            LayerSizes = new();

            while (CurrentLayer > OutputCount)
            {
                LayerSizes.Add(CurrentLayer);

                var CurrentSize = (CurrentLayer / (LayerSizes.Count + 1.0));

                if (CurrentSize < 1)
                {
                    if (LayerSizes.Count > 1)
                        CurrentLayer = OutputCount;
                    else
                        CurrentLayer = (int)Math.Round((CurrentLayer + OutputCount) / 2.0, 0);
                }
                else
                    CurrentLayer = (int)Math.Round(CurrentSize, 0);

                //var fraction = CurrentSize / OutputCount;

                //var multiplier = (int)Math.Round(fraction, 0);

                //if (fraction - (int)fraction == 0.5)
                //{
                //    if (CurrentLayer > OutputCount)
                //        multiplier = (int)fraction;
                //    else
                //        multiplier = (int)(Math.Ceiling(fraction));
                //}

                //CurrentLayer = multiplier * OutputCount;
            }

            HiddenLayers = new();
            int PreviousSize = InputCount;
            foreach (var LayerSize in LayerSizes)
            {
                var NewLayer = new Neuron[LayerSize];
                for (int i = 0; i < LayerSize; i++)
                {
                    NewLayer[i] = new Neuron(PreviousSize);
                }

                HiddenLayers.Add(NewLayer);

                PreviousSize = LayerSize;
            }

            OutputLayer = new Neuron[OutputCount];
            for (int i = 0; i < OutputCount; i++)
                OutputLayer[i] = new Neuron(PreviousSize);

            var r = Random.Shared;
            mutationrate = r.NextSingle();
            mutationintensity = r.NextSingle();
            TrendConfidence = r.NextSingle();
        }

        /// <summary>
        /// Deep cloning an existing NeuralNetwork
        /// </summary>
        public NeuralNetwork(NeuralNetwork other)
        {
            InputCount = other.InputCount;
            OutputCount = other.OutputCount;

            InputAverage = new float[InputCount];
            InputDeviation = new float[InputCount];
            OutputBias = new float[OutputCount];

            for (int x = 0; x < InputCount; x++)
            {
                InputAverage[x] = other.InputAverage[x];
                InputDeviation[x] = other.InputDeviation[x];
            }

            for (int x = 0; x < OutputCount; x++)
                OutputBias[x] = other.OutputBias[x];

            LayerSizes = new();
            HiddenLayers = new();
            for (int LayerIndex = 0; LayerIndex < other.LayerSizes.Count; LayerIndex++)
            {
                var LayerSize = other.LayerSizes[LayerIndex];
                LayerSizes.Add(LayerSize);

                var NewLayer = new Neuron[LayerSize];
                for (int i = 0; i < LayerSize; i++)
                    NewLayer[i] = new Neuron(other.HiddenLayers[LayerIndex][i]);

                HiddenLayers.Add(NewLayer);
            }

            OutputLayer = new Neuron[OutputCount];
            for (int i = 0; i < OutputCount; i++)
                OutputLayer[i] = new Neuron(other.OutputLayer[i]);

            mutationrate = other.mutationrate;
            mutationintensity = other.mutationintensity;
                TrendConfidence = other.TrendConfidence;
        }

        /// <summary>
        /// Run a series of inputs through the nerual network and achieve an output
        /// </summary>
        /// <param name="inputs">The input values</param>
        /// <param name="InputType">If there are multiple input types, this allows input data to be preformatted in different ways</param>
        /// <returns>float[] representing the output of the neural network</returns>
        public float[] GetOutput(float[] inputs)
        {
            //For the purpose of recurrent neural networks, a lower number of inputs is allowed
            var CurrentInputSize = inputs.Length;

            //Preformat inputs with bias and weights
            for (int i = 0; i < CurrentInputSize; i++)
            {
                inputs[i] = (inputs[i] - InputAverage[i]) / InputDeviation[i];
            }

            //Run through every hidden layer
            float[] CurrentOutputs, CurrentInputs;
            CurrentInputs = inputs;

            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                var CurrentSize = LayerSizes[i];
                CurrentOutputs = new float[CurrentSize];

                for (int x = 0; x < CurrentSize; x++)
                    CurrentOutputs[x] = HiddenLayers[i][x].GetOutput(CurrentInputs);

                CurrentInputs = CurrentOutputs;

                CurrentInputSize = CurrentSize;
            }

            //Calculate output layer results
            CurrentOutputs = new float[OutputCount];
            for (int i = 0; i < OutputCount; i++)
            {
                CurrentOutputs[i] = Activation(OutputLayer[i].GetOutput(CurrentInputs, true) + OutputBias[i]);
            }

            return CurrentOutputs;
        }

        /// <summary>
        /// Activation function, sigmoid
        /// </summary>
        /// <param name="d">Value to modify</param>
        /// <returns>modified value</returns>
        float Activation(float f)
        {
            return (1f / (1f + MathF.Exp(-1 * f)));
        }

        /// <summary>
        /// Mutate any float value, for use in model parameters
        /// </summary>
        /// <param name="f">input value</param>
        /// <returns>output value, either original or mutated</returns>
        float MutateValue(float f)
        {
            var r = Random.Shared;

            if (r.NextSingle() < mutationrate)
            {
                IsMutated = true;

                if (f == mutationrate || f == TrendConfidence)
                {
                    float
                        min = Math.Max(f - mutationintensity, 0f),
                        max = Math.Min(f + mutationintensity, 1f),
                        range = max - min;

                    return r.NextSingle() * range + min;
                }
                else if (f == mutationintensity)
                    return Math.Abs(f + r.NextSingle() * 2 - 1);
                else
                    return f + (r.NextSingle() * 2 - 1) * mutationintensity;
            }
            else
                return f;
        }

        /// <summary>
        /// Perform mutation on the model
        /// </summary>
        public void MutateModel()
        {
            //First, we need to set the mutation rate and intensity, as they are necessary for every other step
            mutationrate = MutateValue(mutationrate);
            mutationintensity = MutateValue(mutationintensity);
            TrendConfidence = MutateValue(TrendConfidence);

            for (int i = 0; i < InputCount; i++)
            {
                InputAverage[i] = MutateValue(InputAverage[i]);

                var dev = Math.Abs(MutateValue(InputDeviation[i]));
                if (dev > 0)
                    InputDeviation[i] = dev;
            }

            for (int i = 0; i < OutputCount; i++)
                OutputBias[i] = MutateValue(OutputBias[i]);

            var Neurons = HiddenLayers.SelectMany(x => x).Concat(OutputLayer);
            foreach (var Neuron in Neurons)
            {
                Neuron.Mutate(mutationrate, mutationintensity);
            }

            if (Neurons.Where(x => x.IsMutated).Any())
                IsMutated = true;
        }

        /// <summary>
        /// Breed two neural networks together. After doing this, you should mutate the resulting model.
        /// </summary>
        /// <param name="x">First Model</param>
        /// <param name="y">Second Model</param>
        public NeuralNetwork(NeuralNetwork x, NeuralNetwork y)
        {
            InputCount = x.InputCount;
            OutputCount = x.OutputCount;

            InputAverage = new float[InputCount];
            InputDeviation = new float[InputCount];

            for (int i = 0; i < InputCount; i++)
            {
                InputAverage[i] = Breed(x.InputAverage[i], y.InputAverage[i]);
                InputDeviation[i] = Breed(x.InputDeviation[i], y.InputDeviation[i]);
            }

            OutputBias = new float[OutputCount];
            for (int i = 0; i < OutputCount; i++)
                OutputBias[i] = Breed(x.OutputBias[i], y.OutputBias[i]);

            LayerSizes = new();
            HiddenLayers = new();
            for (int LayerIndex = 0; LayerIndex < x.LayerSizes.Count; LayerIndex++)
            {
                var LayerSize = x.LayerSizes[LayerIndex];
                LayerSizes.Add(LayerSize);

                var NewLayer = new Neuron[LayerSize];
                for (int i = 0; i < LayerSize; i++)
                    NewLayer[i] = x.HiddenLayers[LayerIndex][i] + y.HiddenLayers[LayerIndex][i];

                HiddenLayers.Add(NewLayer);
            }

            OutputLayer = new Neuron[OutputCount];
            for (int i = 0; i < OutputCount; i++)
                OutputLayer[i] = x.OutputLayer[i] + y.OutputLayer[i];

            mutationrate = Breed(x.mutationrate, y.mutationrate);
            mutationintensity = Breed(x.mutationintensity, y.mutationintensity);
            TrendConfidence = Breed(x.TrendConfidence, y.TrendConfidence);
        }

        public static NeuralNetwork operator +(NeuralNetwork x, NeuralNetwork y)
        {
            return new NeuralNetwork(x, y);
        }

        public void IncreaseMutationRate()
        {
            var r = Random.Shared;

            float
                        min = mutationrate,
                        max = Math.Min(mutationrate + mutationintensity, 1f),
                        range = max - min;

            mutationrate = r.NextSingle() * range + min;
        }

        public void IncreaseMutationIntensity()
        {
            var r = Random.Shared;

            float
                        min = mutationintensity,
                        max = mutationintensity + 1,
                        range = max - min;

            mutationintensity = r.NextSingle() * range + min;
        }

        [DataMember]
        double? _error;                                     //Representation of how many incorrect predictions there were, and by how much
        public double? Error                                //Some additional error values may be added as well, to optimize the RatingsModel
        {
            get => _error;
            set
            {
                _error = value;
                OnPropertyChanged(nameof(Error));
            }
        }

        public override int GetHashCode()
        {
            return Error.GetHashCode();
        }

        public override bool Equals(object? obj)
        {
            if (obj is null) return false;

            if (obj is NeuralNetwork model && model.Error == Error)
                return true;
            return false;
        }

        public static bool operator ==(NeuralNetwork x, NeuralNetwork y) => x.Equals(y);
        public static bool operator !=(NeuralNetwork x, NeuralNetwork y) => !x.Equals(y);

        public static bool operator <(NeuralNetwork x, NeuralNetwork y)
        {
            if (x is null)
                return true;
            else if (y is null)
                return false;
            else if (x.Error is null)
                return true;
            else if (y.Error is null)
                return false;
            else
                return x.Error > y.Error;
        }
        public static bool operator >(NeuralNetwork x, NeuralNetwork y)
        {
            if (y is null)
                return true;
            else if (x is null)
                return false;
            else if (y.Error is null)
                return true;
            else if (x.Error is null)
                return false;
            else
                return x.Error < y.Error;
        }

        public int CompareTo(NeuralNetwork? other)
        {
            if (other is null) return -1;

            if (Error is null && other.Error is null) return 0;
            if (Error is null) return 1;
            if (other.Error is null) return -1;

            if (double.IsNaN(Error.Value) && double.IsNaN(other.Error.Value)) return 0;
            if (double.IsNaN(Error.Value)) return 1;
            if (double.IsNaN(other.Error.Value)) return -1;

            return Error.Value.CompareTo(other.Error.Value);
        }

        public ModelParameters GetModelParameters()
        {
            var HiddenLayerWeights = new float[LayerSizes.Count][,]; //x == Layer Size, y == Input Size
            var HiddenLayerBiases = new float[LayerSizes.Count][]; //Each array is Layer Size
            var HiddenLayerOutputBiases = new float[LayerSizes.Count][]; //Same as above

            Parallel.For(0, LayerSizes.Count, i =>
            //for (int i = 0; i < LayerSizes.Count; i++)
            {
                int inputs;

                if (i == 0)
                    inputs = InputCount;
                else
                    inputs = HiddenLayers[i][0].InputSize;

                HiddenLayerWeights[i] = new float[LayerSizes[i], inputs];
                HiddenLayerBiases[i] = new float[LayerSizes[i]];
                HiddenLayerOutputBiases[i] = new float[LayerSizes[i]];

                for (int x = 0; x < LayerSizes[i]; x++)
                {
                    for (int y = 0; y < inputs; y++)
                        HiddenLayerWeights[i][x, y] = HiddenLayers[i][x].Weights[y];

                    for (int j = 0; j < LayerSizes[i]; j++)
                    {
                        HiddenLayerBiases[i][j] = HiddenLayers[i][j].Bias;
                        HiddenLayerOutputBiases[i][j] = HiddenLayers[i][j].OutputBias;
                    }
                }
            });

            int OutputLayerInputs = OutputLayer[0].InputSize;
            var OutputLayerWeights = new float[OutputCount, OutputLayerInputs]; //x == Layer Size, y == Input Size
            var OutputLayerBiases = new float[OutputCount];
            Parallel.For(0, OutputCount, x =>
            {
                for (int y = 0; y < OutputLayerInputs; y++)
                    OutputLayerWeights[x, y] = OutputLayer[x].Weights[y];

                OutputLayerBiases[x] = OutputBias[x];
            });

            return new ModelParameters
            {
                InputAverages = InputAverage,
                InputDeviations = InputDeviation,
                Layer1Weights = HiddenLayerWeights[0],
                Layer2Weights = HiddenLayerWeights[1],
                //Layer3Weights = HiddenLayerWeights[2],
                Layer1Biases = HiddenLayerBiases[0],
                Layer2Biases = HiddenLayerBiases[1],
                //Layer3Biases = HiddenLayerBiases[2],
                Layer1OutputBiases = HiddenLayerOutputBiases[0],
                Layer2OutputBiases = HiddenLayerOutputBiases[1],
                //Layer3OutputBiases = HiddenLayerOutputBiases[2],
                OutputLayerWeights = OutputLayerWeights,
                OutputLayerBiases = OutputLayerBiases
            };
        }

        /// <summary>
        /// Calculates the total number of floating-point parameters in the network.
        /// </summary>
        public int GetFlatParameterCount()
        {
            int count = 0;
            count += InputAverage.Length;
            count += InputDeviation.Length;
            count += OutputBias.Length;

            foreach (var layer in HiddenLayers)
            {
                foreach (var neuron in layer)
                {
                    count += 1; // Bias
                    count += 1; // OutputBias
                    count += neuron.Weights.Length;
                }
            }

            foreach (var neuron in OutputLayer)
            {
                count += 1; // Bias
                count += 1; // OutputBias
                count += neuron.Weights.Length;
            }

            count += 2; // mutationrate and mutationintensity
            return count;
        }

        /// <summary>
        /// Projects the current neural network model towards the specified target parameters.
        /// </summary>
        public NeuralNetwork ProjectModel(float[] TargetParameters)
        {
            int ParameterIndex = 0;
            var projected = new NeuralNetwork(this);

            float ProjectParameter(float current, float target)
            {
                float difference = target - current;
                float adjustment = Math.Clamp(difference * TrendConfidence, -mutationintensity, mutationintensity);
                return current + adjustment;
            }

            for (int i = 0; i < InputAverage.Length; i++)
                projected.InputAverage[i] = ProjectParameter(InputAverage[i], TargetParameters[ParameterIndex++]);

            for (int i = 0; i < InputDeviation.Length; i++)
                projected.InputDeviation[i] = Math.Abs(ProjectParameter(InputDeviation[i], TargetParameters[ParameterIndex++]));

            for (int i = 0; i < OutputBias.Length; i++)
                projected.OutputBias[i] = ProjectParameter(OutputBias[i], TargetParameters[ParameterIndex++]);

            foreach (var layer in projected.HiddenLayers)
            {
                for (int neuronIndex = 0; neuronIndex < layer.Length; neuronIndex++)
                {
                    var currentNeuron = this.HiddenLayers[projected.HiddenLayers.IndexOf(layer)][neuronIndex];
                    var projectedNeuron = layer[neuronIndex];

                    projectedNeuron.Bias = ProjectParameter(currentNeuron.Bias, TargetParameters[ParameterIndex++]);
                    projectedNeuron.OutputBias = ProjectParameter(currentNeuron.OutputBias, TargetParameters[ParameterIndex++]);

                    for (int w = 0; w < projectedNeuron.Weights.Length; w++)
                        projectedNeuron.Weights[w] = ProjectParameter(currentNeuron.Weights[w], TargetParameters[ParameterIndex++]);
                }
            }

            for (int neuronIndex = 0; neuronIndex < projected.OutputLayer.Length; neuronIndex++)
            {
                var currentNeuron = this.OutputLayer[neuronIndex];
                var projectedNeuron = projected.OutputLayer[neuronIndex];

                projectedNeuron.Bias = ProjectParameter(currentNeuron.Bias, TargetParameters[ParameterIndex++]);
                projectedNeuron.OutputBias = ProjectParameter(currentNeuron.OutputBias, TargetParameters[ParameterIndex++]);

                for (int w = 0; w < projectedNeuron.Weights.Length; w++)
                    projectedNeuron.Weights[w] = ProjectParameter(currentNeuron.Weights[w], TargetParameters[ParameterIndex++]);
            }

            projected.mutationrate = ProjectParameter(mutationrate, TargetParameters[ParameterIndex++]);
            projected.mutationintensity = ProjectParameter(mutationintensity, TargetParameters[ParameterIndex++]);

            return projected;
        }

        /// <summary>
        /// Extracts parameters and error statistics from a family tree of models into a flat, cache-friendly array.
        /// </summary>
        public static void ExtractTreeParameters(List<NeuralNetwork> models, float[] flatParameters, double[] errors, out double sumX, out double denominator)
        {
            int n = models.Count;
            sumX = 0;
            double sumX2 = 0;

            // 1. Extract Errors and calculate the regression denominator
            for (int m = 0; m < n; m++)
            {
                double err = models[m].Error!.Value;
                errors[m] = err;
                sumX += err;
                sumX2 += err * err;
            }

            denominator = (n * sumX2) - (sumX * sumX);

            int paramIndex = 0;

            // 2. Extract InputAverages
            for (int i = 0; i < models[0].InputAverage.Length; i++)
            {
                for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].InputAverage[i];
                paramIndex++;
            }

            // 3. Extract InputDeviations
            for (int i = 0; i < models[0].InputDeviation.Length; i++)
            {
                for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].InputDeviation[i];
                paramIndex++;
            }

            // 4. Extract OutputBias
            for (int i = 0; i < models[0].OutputBias.Length; i++)
            {
                for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].OutputBias[i];
                paramIndex++;
            }

            // 5. Extract Hidden Layers
            for (int layerIndex = 0; layerIndex < models[0].HiddenLayers.Count; layerIndex++)
            {
                var layer = models[0].HiddenLayers[layerIndex];
                for (int neuronIndex = 0; neuronIndex < layer.Length; neuronIndex++)
                {
                    // Bias
                    for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].HiddenLayers[layerIndex][neuronIndex].Bias;
                    paramIndex++;

                    // OutputBias
                    for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].HiddenLayers[layerIndex][neuronIndex].OutputBias;
                    paramIndex++;

                    // Weights
                    int weightCount = layer[neuronIndex].Weights.Length;
                    for (int w = 0; w < weightCount; w++)
                    {
                        for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].HiddenLayers[layerIndex][neuronIndex].Weights[w];
                        paramIndex++;
                    }
                }
            }

            // 6. Extract Output Layer
            for (int neuronIndex = 0; neuronIndex < models[0].OutputLayer.Length; neuronIndex++)
            {
                // Bias
                for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].OutputLayer[neuronIndex].Bias;
                paramIndex++;

                // OutputBias
                for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].OutputLayer[neuronIndex].OutputBias;
                paramIndex++;

                // Weights
                int weightCount = models[0].OutputLayer[neuronIndex].Weights.Length;
                for (int w = 0; w < weightCount; w++)
                {
                    for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].OutputLayer[neuronIndex].Weights[w];
                    paramIndex++;
                }
            }

            // 7. Extract Mutation Parameters
            for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].mutationrate;
            paramIndex++;

            for (int m = 0; m < n; m++) flatParameters[(paramIndex * n) + m] = models[m].mutationintensity;
        }

        /// <summary>
        /// Calculates the optimal target parameters via linear regression, converting double-precision math back to floats.
        /// </summary>
        public static void CalculateTrendParameters(float[] inputParameters, float[] targetParameters, double[] errors, int paramCount, int modelCount, double sumX, double denominator)
        {
            Parallel.For(0, paramCount, p =>
            {
                int startIndex = p * modelCount;

                // CRITICAL: Accumulators MUST be double to prevent floating point overflow!
                double sumY = 0, sumXY = 0;

                for (int m = 0; m < modelCount; m++)
                {
                    // Read the float, but do math in double-precision
                    double yValue = inputParameters[startIndex + m];
                    sumY += yValue;
                    sumXY += errors[m] * yValue;
                }

                if (Math.Abs(denominator) < 1e-10)
                {
                    targetParameters[p] = (float)(sumY / modelCount);
                }
                else
                {
                    double slope = (modelCount * sumXY - sumX * sumY) / denominator;
                    targetParameters[p] = (float)((sumY - slope * sumX) / modelCount);
                }
            });
        }

        public record ModelStats(int Layer1Size, int Layer2Size, int Layer3Size, int Layer1Inputs, int Layer2Inputs, int Layer3Inputs, int Outputs, int OutputInputs, int MemorySize);

        public ModelStats GetModelStats()
        {
            int
                Layer1Size = HiddenLayers.Count > 0 ? HiddenLayers[0].Length : 0,
                Layer2Size = HiddenLayers.Count > 1 ? HiddenLayers[1].Length : 0,
                Layer3Size = HiddenLayers.Count > 2 ? HiddenLayers[2].Length : 0,
                Layer1Inputs = InputCount,
                Layer2Inputs = Layer2Size > 0 ? HiddenLayers[1][0].InputSize : 0,
                Layer3Inputs = Layer3Size > 0 ? HiddenLayers[2][0].InputSize : 0,
                Outputs = OutputCount,
                OutputInputs = Outputs > 0 ? OutputLayer[0].InputSize : 0,
                MemorySize =
                    Layer1Inputs * 2 +
                    Layer1Size * 2 +
                    Layer2Size * 2 +
                    Layer3Size * 2 +
                    Outputs +
                    Layer1Size * Layer1Inputs +
                    Layer2Size * Layer2Inputs +
                    Layer3Size * Layer3Inputs +
                    Outputs * OutputInputs;

            MemorySize *= 4;

            return new ModelStats(Layer1Size, Layer2Size, Layer3Size, Layer1Inputs, Layer2Inputs, Layer3Inputs, Outputs, OutputInputs, MemorySize);                
            
        }
    }
}
