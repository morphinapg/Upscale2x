using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace Upscale2x.ViewModels
{
    [DataContract]
    public class Neuron : Breedable
    {
        [DataMember]
        public float Bias = 0, OutputBias = 0;

        [DataMember]
        public float[] Weights;

        [DataMember]
        public int InputSize;

        [DataMember]
        public bool IsMutated = false;

        /// <summary>
        /// Initialize Neuron with just InputSize
        /// </summary>
        /// <param name="inputs">The number of inputs</param>
        public Neuron(int inputs)
        {
            var r = Random.Shared;
            Weights = new float[inputs];
            //for (int i = 0; i < inputs; i++)
            //    weights[i] = r.NextDouble() * 2 - 1;

            //bias = r.NextDouble() * 2 - 1;
            //outputbias = r.NextDouble() * 2 - 1;
            InputSize = inputs;
        }

        /// <summary>
        /// Clone existing Neuron
        /// </summary>
        /// <param name="other">Neuron to clone</param>
        public Neuron(Neuron other)
        {
            Bias = other.Bias;
            OutputBias = other.OutputBias;
            InputSize = other.InputSize;

            Weights = new float[InputSize];
            for (int i = 0; i < InputSize; i++)
                Weights[i] = other.Weights[i];
        }

        /// <summary>
        /// Breed two neurons together
        /// </summary>
        /// <param name="x">First Neuron</param>
        /// <param name="y">Second Neuron</param>
        public Neuron(Neuron x, Neuron y)
        {
            var r = Random.Shared;
            Bias = Breed(x.Bias, y.Bias);
            OutputBias = Breed(x.OutputBias, y.OutputBias);

            InputSize = x.InputSize;

            Weights = new float[x.InputSize];
            for (int i = 0; i < InputSize; i++)
                Weights[i] = Breed(x.Weights[i], y.Weights[i]);
        }

        public static Neuron operator +(Neuron x, Neuron y)
        {
            return new Neuron(x, y);
        }

        /// <summary>
        /// Activation function
        /// </summary>
        /// <param name="d">Value to modify</param>
        /// <returns>modified value</returns>
        float Activation(float f)
        {
            return (2f / (1f + MathF.Exp(-1 * f))) - 1;
        }

        /// <summary>
        /// Produce an output given a series of inputs
        /// </summary>
        /// <param name="inputs">Inputs for the Neuron to process</param>
        /// <param name="output">Whether the Neuron is on the output layer or not</param>
        /// <returns>The output of the neuron's processing</returns>
        public float GetOutput(float[] inputs, bool output = false)
        {
            float total = 0;

            for (int i = 0; i < inputs.Length; i++)
                total += inputs[i] * Weights[i];

            total += Bias;

            return output ? total : Activation(total) + OutputBias;
        }

        /// <summary>
        /// Mutate the neuron
        /// </summary>
        /// <param name="mutationrate">The mutation rate</param>
        /// <param name="mutationintensity">The mutation intensity</param>
        public void Mutate(float mutationrate, float mutationintensity)
        {
            var r = Random.Shared;

            for (int i = 0; i < InputSize; i++)
                if (r.NextSingle() < mutationrate)
                {
                    Weights[i] += mutationintensity * (r.NextSingle() * 2 - 1);
                    IsMutated = true;
                }

            if (r.NextSingle() < mutationrate)
            {
                Bias += mutationintensity * (r.NextSingle() * 2 - 1);
                IsMutated = true;
            }

            if (r.NextSingle() < mutationrate)
            {
                float
                    min = MathF.Max(OutputBias - mutationintensity, 0f),
                    max = MathF.Min(OutputBias + mutationintensity, 1f);

                var bias = r.NextSingle() * (max - min) + min;

                OutputBias = bias;
                IsMutated = true;
            }
        }
    }
}
