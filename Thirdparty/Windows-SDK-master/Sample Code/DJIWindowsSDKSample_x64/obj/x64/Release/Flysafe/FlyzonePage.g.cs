﻿#pragma checksum "C:\Users\artur\Documents\DJI Sample\Windows-SDK-master\Sample Code\DJISampleSources\Flysafe\FlyzonePage.xaml" "{406ea660-64cf-4c82-b6f0-42d48172a799}" "E03CF88074EE30925EC1E9EDE1CD7E04"
//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace DJIWindowsSDKSample.Flysafe
{
    partial class FlyzonePage : 
        global::Windows.UI.Xaml.Controls.Page, 
        global::Windows.UI.Xaml.Markup.IComponentConnector,
        global::Windows.UI.Xaml.Markup.IComponentConnector2
    {
        /// <summary>
        /// Connect()
        /// </summary>
        [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.Windows.UI.Xaml.Build.Tasks"," 10.0.17.0")]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public void Connect(int connectionId, object target)
        {
            switch(connectionId)
            {
            case 2: // Flysafe\FlyzonePage.xaml line 42
                {
                    this.FlyZoneMap = (global::Windows.UI.Xaml.Controls.Maps.MapControl)(target);
                }
                break;
            case 3: // Flysafe\FlyzonePage.xaml line 30
                {
                    this.simulatButton = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.simulatButton).Click += this.simulatButton_Click;
                }
                break;
            case 4: // Flysafe\FlyzonePage.xaml line 34
                {
                    this.showFlyzoneButton = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.showFlyzoneButton).Click += this.showFlyzoneButton_Click;
                }
                break;
            default:
                break;
            }
            this._contentLoaded = true;
        }

        /// <summary>
        /// GetBindingConnector(int connectionId, object target)
        /// </summary>
        [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.Windows.UI.Xaml.Build.Tasks"," 10.0.17.0")]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public global::Windows.UI.Xaml.Markup.IComponentConnector GetBindingConnector(int connectionId, object target)
        {
            global::Windows.UI.Xaml.Markup.IComponentConnector returnValue = null;
            return returnValue;
        }
    }
}

