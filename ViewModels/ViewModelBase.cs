using CommunityToolkit.Mvvm.ComponentModel;
using System.Runtime.Serialization;

namespace Upscale2x.ViewModels
{
    [DataContract(IsReference = true)]
    public class ViewModelBase : ObservableObject
    {
    }
}
