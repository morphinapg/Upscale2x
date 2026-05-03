using Avalonia.Controls;
using Avalonia.Interactivity;
using Upscale2x.ViewModels;

namespace Upscale2x.Views
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        protected override void OnLoaded(RoutedEventArgs e)
        {
            base.OnLoaded(e);

            CurrentApp.TopLevel = TopLevel.GetTopLevel(MainGrid);
        }
    }
}