using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace Upscale2x.ViewModels
{
    [DataContract]
    public class Breedable : ViewModelBase
    {
        /// <summary>
        /// Random blend between two values
        /// </summary>
        /// <param name="x">first value</param>
        /// <param name="y">second value</param>
        /// <returns></returns>
        public float Breed(float x, float y)
        {
            var r = Random.Shared;
            var p = r.NextSingle();

            return (x * p) + (y * (1 - p));
        }

        public float4 Breed(float4 x, float4 y)
        {
            return new float4(
                Breed(x.X, y.X), 
                Breed(x.Y, y.Y), 
                Breed(x.Z, y.Z), 
                Breed(x.W, y.W)
                );
        }
    }
}
