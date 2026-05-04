using Avalonia.Controls;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Avalonia.SimplePreferences;
using ComputeSharp;
using ComputeSharp.Resources;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
using System.Timers;
using Tmds.DBus.Protocol;

namespace Upscale2x.ViewModels
{
    public partial class MainViewModel : ViewModelBase, IDisposable
    {
        Bitmap? _inputImageBitmap, _outputImage;        

        public Bitmap? InputImageBitmap
        {
            get => _inputImageBitmap;
            set
            {
                _inputImageBitmap = value;
                OnPropertyChanged(nameof(InputImageBitmap));
            }
        }
        public Bitmap? OutputImage
        {
            get => _outputImage;
            set
            {
                _outputImage = value;
                OnPropertyChanged(nameof(OutputImage));
            }
        }

        Evolution? _baseModel;
        public Evolution? BaseModel
        {
            get => _baseModel;
            set
            {
                _baseModel = value;
                OnPropertyChanged(nameof(BaseModel));
                OnPropertyChanged(nameof(BaseModelEnabled));
                OnPropertyChanged(nameof(BaseImprovement));
                OnPropertyChanged(nameof(TotalImprovement));
                OnPropertyChanged(nameof(TotalVisible));
            }
        }

        Evolution? _refineModel;
        public Evolution? RefineModel
        {
            get => _refineModel;
            set
            {
                _refineModel = value;
                OnPropertyChanged(nameof(RefineModel));
                OnPropertyChanged(nameof(RefineModelEnabled));
                OnPropertyChanged(nameof(RefineImprovement));
                OnPropertyChanged(nameof(TotalImprovement));
                OnPropertyChanged(nameof(TotalVisible));
            }
        }

        /// <summary>
        /// This will store the error of the base model before training starts, so we can compare it to the current error and see the improvement.
        /// </summary>
        double? _baseError;
        public double? BaseError
        { 
            get => _baseError;
            set
            {
                _baseError = value;
                OnPropertyChanged(nameof(BaseError));
            }
        }

        /// <summary>
        /// Gets the relative improvement of the model error compared to the base error, if a base model is available.
        /// </summary>
        /// <remarks>The improvement is calculated as (BaseError - BaseModel.Error) / BaseError. If no
        /// base model is present, the property returns null.</remarks>
        public double? BaseImprovement
        {
            get
            {
                if (BaseModel is not null)
                {
                    return (BaseError - BaseModel.Error) / BaseError;
                }
                else
                    return null;
            }
        }

        /// <summary>
        /// Gets the relative improvement in error achieved by the refined model compared to the base model.
        /// </summary>
        /// <remarks>Returns null if the refined model is not available. The improvement is calculated as
        /// the fractional reduction in error from the base model to the refined model. A positive value indicates a
        /// reduction in error, while a negative value indicates an increase.</remarks>
        public double? RefineImprovement
        {
            get
            {
                if (RefineModel is not null)
                {
                    if (BaseModel is not null && BaseModel.Error is not null)
                    {
                        return (BaseModel.Error - RefineModel.Error) / BaseModel.Error;
                    }
                    else
                        return (BaseError - RefineModel.Error) / BaseError;
                }
                else
                    return null;
            }
        }

        public double? TotalImprovement
        {
            get
            {
                if (BaseModel is not null && RefineModel is not null && BaseModel.Error is not null && RefineModel.Error is not null && BaseError is not null)
                {
                    return (BaseError - RefineModel.Error) / BaseError;
                }
                else
                    return null;
            }
        }

        public bool TotalVisible => TotalImprovement is not null;

        public bool BaseModelEnabled => BaseModel is not null;
        public bool RefineModelEnabled => RefineModel is not null;

        public bool ModelEnabled => BaseModelEnabled || RefineModelEnabled;

        bool _isTraining = false;
        bool IsTraining
        {
            get => _isTraining;
            set
            {
                _isTraining = value;
                OnPropertyChanged(nameof(TrainingMessage));
                OnPropertyChanged(nameof(IsNotTraining));
            }
        }

        public bool IsNotTraining => !IsTraining;

        //bool _acceleratedMode = true;
        //public bool AcceleratedMode
        //{
        //    get => _acceleratedMode;
        //    set
        //    {
        //        _acceleratedMode = value;
        //        Evolution?.ResetErrors();

        //        OnPropertyChanged(nameof(AcceleratedMode));

        //        if (Evolution is not null)
        //            ModelReset = RefineMode != AcceleratedMode;
        //    }
        //}

        bool _modelReset = false;
        public bool ModelReset
        {
            get => _modelReset;
            set
            {
                _modelReset = value;
                OnPropertyChanged(nameof(ModelReset));
            }
        }

        bool _analyzeRender = false;
        public bool AnalyzeRender
        {
            get => _analyzeRender;
            set
            {
                _analyzeRender = value;
                OnPropertyChanged(nameof(AnalyzeRender));
            }
        }

        bool _refineMode = false;
        public bool RefineMode
        {
            get => _refineMode;
            set
            {
                _refineMode = value;
                OnPropertyChanged(nameof(RefineMode));
                OnPropertyChanged(nameof(BaseMode));

                if (value)
                {
                    if (BaseModel is not null)
                    {
                        //We need to check whether the base model has been trained and beats the base error
                        //If not, we need to invalidate the base model
                        //refine model will then be based on a bilinear upscale as a base instead

                        if (BaseModel.Error is null || BaseModel.Error >= BaseError)
                        {
                            BaseModel = null;   
                            OnPropertyChanged(nameof(BaseModelEnabled));
                            OnPropertyChanged(nameof(BaseImprovement));
                            OnPropertyChanged(nameof(TotalImprovement));
                        }
                    }

                    if (RefineModel is null)
                    {
                        RefineModel = new Evolution(true);

                        Task.Run(() =>
                        {
                            RefineModel.NextGeneration(InputImage, DownscaledImage, true, BaseModel?.TopModel);

                            OnPropertyChanged(nameof(RefineModelEnabled));
                            OnPropertyChanged(nameof(RefineImprovement));
                            OnPropertyChanged(nameof(TotalImprovement));
                        });
                    }
                }
                else
                {
                    if (BaseModel is null)
                    {
                        BaseModel = new Evolution(false);

                        Task.Run(() =>
                        {
                            BaseModel.NextGeneration(InputImage, DownscaledImage, false);

                            OnPropertyChanged(nameof(BaseModelEnabled));
                            OnPropertyChanged(nameof(BaseImprovement));
                            OnPropertyChanged(nameof(RefineImprovement));
                            OnPropertyChanged(nameof(TotalImprovement));
                        });
                    }

                    if (RefineModel is not null)
                    {
                        //We need to check whether the refine model has been trained and beats the Base Model's error
                        //If not, we need to invalidate the refine model

                        if (RefineModel.Error is null || (RefineModel.Error >= BaseModel.Error))
                        {
                            RefineModel = null;
                            OnPropertyChanged(nameof(RefineModelEnabled));
                            OnPropertyChanged(nameof(RefineImprovement));
                            OnPropertyChanged(nameof(TotalImprovement));
                        }
                    }
                }
            }
        }

        public bool BaseMode => RefineMode == false;

        public string TrainingMessage => IsTraining ? "Stop Training" : "Start Training";

        System.Timers.Timer StatusTimer = new();

        public string? TrainingStatus
        {
            get
            {
                var Model = RefineMode ? RefineModel : BaseModel;

                if (Model is null || Error is null)
                    return null;
                else
                {
                    string message =
                        "Generations: " + Generations + "\r\n" +
                        "Current Error: " + Error + "\r\n";

                    if (OldError is not null)
                    {
                        var Improvement = 1 - Error.Value / OldError.Value;

                        message +=
                           "Original Error: " + OldError + "\r\n" +
                           "Improvement: " + Improvement.ToString("P2") + "\r\n";
                    }
                    else
                        Model.OldError = Error;
                       

                    message += Model.LastUpdateText;

                    return message;
                }
            }
        }

        string? _openPath;
        string? OpenPath
        {
            get
            {
                if (_openPath is null && Preferences.Get("OpenPath", _openPath) is string StoredPath)
                {
                    _openPath = StoredPath;
                }
                return _openPath;
            }
            set
            {
                _openPath = value;
                Preferences.Set("OpenPath", value);
            }
        }

        public CommandHandler Load_Image => new CommandHandler(LoadImage);
        async void LoadImage()
        {
            var toplevel = CurrentApp.TopLevel;
            if (toplevel is not null && toplevel.StorageProvider is not null)
            {
                var ImageTypes = new FilePickerFileType("Image Files");
                ImageTypes.Patterns = new[] { "*.png", "*.jpg", "*.tif", "*.bmp", "*.jpeg" };

                var options = new FilePickerOpenOptions
                {
                    Title = "Open Image",
                    AllowMultiple = false,
                    FileTypeFilter = new List<FilePickerFileType> { ImageTypes }
                };

                if (OpenPath is not null)
                    options.SuggestedStartLocation = await toplevel.StorageProvider.TryGetFolderFromPathAsync(OpenPath);

                var file = await toplevel.StorageProvider.OpenFilePickerAsync(options);

                if (file is not null && file.Any())
                {
                    var path = file.First().TryGetLocalPath();

                    OpenPath = Path.GetDirectoryName(path);

                    if (path is not null)
                    {
                        InputImageBitmap = new Bitmap(path);

                        await PrepareImage(path);
                    }
                }
            }
        }

        ReadWriteTexture2D<Rgba64, float4>? _downscaledTexture;
        public ReadWriteTexture2D<Rgba64, float4>? DownscaledImage
        {
            get => _downscaledTexture;
            set
            {
                _downscaledTexture?.Dispose();

                _downscaledTexture = value;
            }
        }

        ReadWriteTexture2D<Rgba64, float4>? _inputImage;
        public ReadWriteTexture2D<Rgba64, float4>? InputImage
        {
            get => _inputImage;
            set
            {
                _inputImage?.Dispose();

                _inputImage = value;
            }
        }

        public void Dispose()
        {
            _downscaledTexture?.Dispose();
            _inputImage?.Dispose();
        }

        public bool ReadyToTrain => ModelEnabled && DownscaledImage is not null && InputImage is not null;

        async Task PrepareImage(string path)
        {
            await Task.Run(() =>
            {
                var GPU = GraphicsDevice.GetDefault();
                if (InputImage is not null)
                    InputImage.Dispose();
                InputImage = GPU.LoadReadWriteTexture2D<Rgba64, float4>(path);
                var TotalPixels = InputImage.Width * InputImage.Height;

                if (DownscaledImage is not null)
                    DownscaledImage.Dispose();
                DownscaledImage = GPU.AllocateReadWriteTexture2D<Rgba64, float4>(InputImage.Width / 2, InputImage.Height / 2);

                GPU.ForEach(DownscaledImage, new GPUDownscale(InputImage));       

                if (BaseMode && BaseModel is null)
                    BaseModel = new Evolution(false);
                
                if (RefineMode && RefineModel is null)
                    RefineModel = new Evolution(true);                

                OnPropertyChanged(nameof(BaseModelEnabled));
                OnPropertyChanged(nameof(RefineModelEnabled));
                OnPropertyChanged(nameof(ModelEnabled));
                OnPropertyChanged(nameof(ReadyToTrain));

                RefreshModelErrors();
            });            
        }

        void RefreshModelErrors()
        {
            Task.Run(() =>
            {
                //We need to calculate BaseError, which will be used as reference for the total model improvement
                //For this we will create a temporary Evolution instance and calculate the error without any training, as this will be the error of a bilinear upscale which is our baseline for improvement
                var TempEvolution = new Evolution(false);
                BaseError = TempEvolution.InitializeError(InputImage, DownscaledImage, false);

                //Now we need to update the errors of the current models, as they might be based on a different image or have been trained in refine mode with a different base model, so their errors are not comparable to the current base error
                if (BaseModel is not null)
                {
                    BaseModel.ResetErrors();
                    BaseModel.NextGeneration(InputImage, DownscaledImage, false);
                }
                if (RefineModel is not null)
                {
                    RefineModel.ResetErrors();
                    RefineModel.NextGeneration(InputImage, DownscaledImage, true, BaseModel?.TopModel);
                }

                //After refreshing the errors, we need to notify the UI to update the improvement values
                OnPropertyChanged(nameof(BaseImprovement));
                OnPropertyChanged(nameof(RefineImprovement));
                OnPropertyChanged(nameof(TotalImprovement));
                OnPropertyChanged(nameof(TotalVisible));
            });            
        }

        public CommandHandler Reset_Model => new CommandHandler(ResetModel);

        void ResetModel()
        {
            ////if (AcceleratedMode)
            //    BaseModel?.ResetEvolution();
            ////else if (Average.HasValue && Deviation.HasValue)
            //    //Evolution?.ResetEvolution(Average.Value, Deviation.Value);

            ////RefineMode = AcceleratedMode;

            //if (BaseModel is not null)
            //    BaseModel.RefineModel = RefineMode;

            //ModelReset = false;

            //OnPropertyChanged(nameof(ModelEnabled));
            //OnPropertyChanged(nameof(ReadyToTrain));

            if (RefineMode)
            {
                if (RefineModel is not null)
                    RefineModel.ResetEvolution();
                else
                    RefineModel = new Evolution(true);

                BaseModel = null;

                OnPropertyChanged(nameof(BaseModelEnabled));
                OnPropertyChanged(nameof(BaseImprovement));

                OnPropertyChanged(nameof(RefineImprovement));
                OnPropertyChanged(nameof(TotalImprovement));
                OnPropertyChanged(nameof(TotalVisible));
            }
            else
            {
                if (BaseModel is not null)
                    BaseModel.ResetEvolution();
                else
                    BaseModel = new Evolution(false);

                RefineModel = null;
                OnPropertyChanged(nameof(RefineModelEnabled));
                OnPropertyChanged(nameof(RefineImprovement));
                OnPropertyChanged(nameof(TotalImprovement));
                OnPropertyChanged(nameof(TotalVisible));
                OnPropertyChanged(nameof(BaseImprovement));
            }
        }

        public MainViewModel()
        {
            StatusTimer.Elapsed += (s, e) => OnPropertyChanged(nameof(TrainingStatus));
        }

        public CommandHandler StartStop_Training => new CommandHandler(StartStopTraining);
        async void StartStopTraining()
        {
            var Model = RefineMode ? RefineModel : BaseModel;

            if (Model is not null)
            {
                if (IsTraining)
                {
                    IsTraining = false;

                    OnPropertyChanged(nameof(BaseImprovement));                    

                    if (BaseMode && RefineModel is not null && BaseModel is not null && RefineModel.Error > BaseModel.Error)
                    {
                        RefineModel = null;
                        OnPropertyChanged(nameof(RefineModelEnabled));
                    }

                    OnPropertyChanged(nameof(RefineImprovement));
                    OnPropertyChanged(nameof(TotalImprovement));
                    OnPropertyChanged(nameof(TotalVisible));
                }
                else
                {
                    IsTraining = true;

                    StatusTimer.Start();

                    Model.OldError = Model.Error;
                    Model.ResetSecondary = true;

                    await Task.Run(() =>
                    {
                        while (IsTraining)
                        {
                            var basemodel = RefineMode ? BaseModel?.TopModel : null;

                            Model.NextGeneration(InputImage, DownscaledImage, RefineMode, basemodel);

                            OnPropertyChanged(nameof(BaseImprovement));
                            OnPropertyChanged(nameof(RefineImprovement));
                            OnPropertyChanged(nameof(TotalImprovement));
                            OnPropertyChanged(nameof(TotalVisible));
                        }
                    });
                }
            }            
        }

        public double? Error// => BaseModel is not null ? BaseModel.Error : null;
        {
            get
            {
                var model = RefineMode ? RefineModel : BaseModel;
                return model is not null ? model.Error : null;
            }
        }
        public double? OldError //BaseModel is not null ? BaseModel.OldError : null;
        {
            get
            {
                var model = RefineMode ? RefineModel : BaseModel;
                return model is not null ? model.OldError : null;
            }
        }

        public int? Generations// => BaseModel is not null ? BaseModel.Generations : null;
        {
            get
            {
                var model = RefineMode ? RefineModel : BaseModel;
                return model is not null ? model.Generations : null;
            }
        }

        SKBitmap? Bitmap16;


        public CommandHandler Upscale_Click => new CommandHandler(Upscale);
        async void Upscale()
        {
            var Model = RefineMode ? RefineModel : this.BaseModel;
            if (Model is not null)
            {
                var basemodel = RefineMode ? BaseModel?.TopModel : null;

                var outputs = await Model.Upscale(InputImage, RefineMode, AnalyzeRender, basemodel);

                if (outputs is not null)
                {
                    OutputImage = outputs.Bitmap8;

                    Bitmap16 = outputs.Bitmap16;
                }

            }
        }

        string? _savePath;
        string? SavePath
        {
            get
            {
                if (_savePath is null && Preferences.Get("SavePath", _savePath) is string StoredPath)
                {
                    _savePath = StoredPath;
                }

                return _savePath;
            }
            set
            {
                _savePath = value;
                Preferences.Set("SavePath", value);
            }
        }

        public CommandHandler Save_Click => new CommandHandler(SaveToFile);

        async void SaveToFile()
        {
            var toplevel = CurrentApp.TopLevel;
            if (toplevel is not null && toplevel.StorageProvider is not null && Bitmap16 is not null)
            {
                var FileTypes = new List<FilePickerFileType>();
                FileTypes.Add(new FilePickerFileType("16-bit PNG") { Patterns = new[] { "*.png" } });
                FileTypes.Add(new FilePickerFileType("JPEG") { Patterns = new[] { "*.jpg" } });

                var options = new FilePickerSaveOptions
                {
                    Title = "Save Image",
                    FileTypeChoices = FileTypes
                };

                if (SavePath is not null)
                    options.SuggestedStartLocation = await toplevel.StorageProvider.TryGetFolderFromPathAsync(SavePath);

                var file = await toplevel.StorageProvider.SaveFilePickerAsync(options);

                if (file is not null)
                {
                    var path = file.TryGetLocalPath();

                    if (path is not null)
                    {
                        SavePath = Path.GetDirectoryName(path);

                        using (var stream = await file.OpenWriteAsync())
                        {
                            var extension = Path.GetExtension(path).ToLower();

                            if (extension  == ".jpg")
                                Bitmap16.Encode(stream, SKEncodedImageFormat.Jpeg, 100);
                            else
                                Bitmap16.Encode(stream, SKEncodedImageFormat.Png, 100);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Write an object to file
        /// </summary>
        /// <typeparam name="T">Type of object to write</typeparam>
        /// <param name="FileName">File Name</param>
        /// <param name="item">The object to write</param>
        async Task WriteObjectAsync<T>(string FileName, T item)
        {
            await Task.Run(() =>
            {
                using (var writer = new FileStream(FileName, FileMode.Create))
                {
                    new DataContractSerializer(typeof(T)).WriteObject(writer, item);
                }
            });
        }

        /// <summary>
        /// Save an object to file
        /// </summary>
        /// <typeparam name="T">Type of object to save</typeparam>
        /// <param name="FileName">File name</param>
        async Task<T?> ReadObjectAsync<T>(string FileName)
        {
            var item = await Task.Run(() =>
            {
                using (var fs = new FileStream(FileName, FileMode.Open, FileAccess.Read))
                {
                    return new DataContractSerializer(typeof(T)).ReadObject(fs);
                }
            });

            if (item is null) return default;

            return (T)item;
        }

        string? _modelPath;
        string? ModelPath
        {
            get
            {
                if (_modelPath is null && Preferences.Get("ModelPath", _modelPath) is string StoredPath)
                {
                    _modelPath = StoredPath;
                }
                return _modelPath;
            }
            set
            {
                _modelPath = value;
                Preferences.Set("ModelPath", value);
            }
        }

        public CommandHandler SaveModel_Click => new CommandHandler(SaveModel);

        async void SaveModel()
        {
            var toplevel = CurrentApp.TopLevel;
            if (toplevel is not null && toplevel.StorageProvider is not null && BaseModel is not null)
            {
                var FileTypes = new List<FilePickerFileType>();
                FileTypes.Add(new FilePickerFileType("AI Model") { Patterns = new[] { "*.model" } });

                var options = new FilePickerSaveOptions
                {
                    Title = "Save Model",
                    FileTypeChoices = FileTypes
                };

                if (ModelPath is not null)
                    options.SuggestedStartLocation = await toplevel.StorageProvider.TryGetFolderFromPathAsync(ModelPath);

                var file = await toplevel.StorageProvider.SaveFilePickerAsync(options);

                if (file is not null)
                {
                    var path = file.TryGetLocalPath();

                    if (path is not null)
                    {
                        ModelPath = Path.GetDirectoryName(path);

                        Evolution? Base = null, Refine = null;

                        if (BaseModel is not null)
                            Base = new Evolution(BaseModel);

                        if (RefineModel is not null)
                            Refine = new Evolution(RefineModel);

                        var Bundle = new ModelBundle
                        {
                            BaseModel = Base,
                            RefineModel = Refine,
                            RefineMode = RefineMode
                        };

                        await WriteObjectAsync(path, Bundle);
                    }
                }
            }
        }

        public CommandHandler LoadModel_Click => new CommandHandler(LoadModel);

        async void LoadModel()
        {
            var toplevel = CurrentApp.TopLevel;
            if (toplevel is not null && toplevel.StorageProvider is not null)
            {
                var FileTypes = new List<FilePickerFileType>();
                FileTypes.Add(new FilePickerFileType("AI Model") { Patterns = new[] { "*.model" } });

                var options = new FilePickerOpenOptions
                {
                    Title = "Open Image",
                    AllowMultiple = false,
                    FileTypeFilter = FileTypes
                };

                if (ModelPath is not null)
                    options.SuggestedStartLocation = await toplevel.StorageProvider.TryGetFolderFromPathAsync(ModelPath);

                var file = await toplevel.StorageProvider.OpenFilePickerAsync(options);

                if (file is not null && file.Any())
                {
                    var path = file.First().TryGetLocalPath();

                    if (path is not null)
                    {
                        ModelPath = Path.GetDirectoryName(path);

                        var model = await ReadObjectAsync<ModelBundle>(path);

                        if (model is not null)
                        {
                            BaseModel = model.BaseModel;
                            RefineModel = model.RefineModel;
                            RefineMode = model.RefineMode;

                            //if (InputImage is not null && DownscaledImage is not null)
                            //    UpdateImages(InputImage, DownscaledImage);
                        }

                        OnPropertyChanged(nameof(ModelEnabled));
                        OnPropertyChanged(nameof(ReadyToTrain));
                        OnPropertyChanged(nameof(RefineMode));

                        RefreshModelErrors();
                    }
                }
            }
        }

        public CommandHandler ResetBaseModel_Click => new CommandHandler(ResetBaseModel);

        void ResetBaseModel()
        {
            if (RefineMode)
            {
                BaseModel = null;

                OnPropertyChanged(nameof(BaseModelEnabled));
                OnPropertyChanged(nameof(BaseImprovement));

                OnPropertyChanged(nameof(RefineImprovement));
                OnPropertyChanged(nameof(TotalImprovement));
                OnPropertyChanged(nameof(TotalVisible));
            }
            else
            {
                if (BaseModel is not null)
                    BaseModel.ResetEvolution();
                else
                    BaseModel = new Evolution(false);

                OnPropertyChanged(nameof(RefineModelEnabled));
                OnPropertyChanged(nameof(RefineImprovement));
                OnPropertyChanged(nameof(TotalImprovement));
                OnPropertyChanged(nameof(TotalVisible));
                OnPropertyChanged(nameof(BaseImprovement));
            }
        }

        public CommandHandler ResetRefineModel_Click => new CommandHandler(ResetRefineModel);

        void ResetRefineModel()
        {
            if (RefineMode)
            {
                if (RefineModel is not null)
                    RefineModel.ResetEvolution();
                else
                    RefineModel = new Evolution(true);

                OnPropertyChanged(nameof(RefineModelEnabled));
                OnPropertyChanged(nameof(RefineImprovement));
                OnPropertyChanged(nameof(TotalImprovement));
                OnPropertyChanged(nameof(TotalVisible));
            }
            else
            {
                RefineModel = null;
                OnPropertyChanged(nameof(RefineModelEnabled));
                OnPropertyChanged(nameof(RefineImprovement));
                OnPropertyChanged(nameof(TotalImprovement));
                OnPropertyChanged(nameof(TotalVisible));
            }
        }
    }
}
