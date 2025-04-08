在Time Series Library (TSlib)的项目基础上实现的。

原README.md地址:https://github.com/thuml/Time-Series-Library

这里没有把打包好的runGUI.exe都一块放进去,否则整个项目过于庞大了。

如果需要打包exe，需要下载pyinstaller，打开Terminal，在项目根目录下执行:
```shell
pyinstaller --onefile --windowed --add-data="convert_paths;convert_paths" --add-data="data_provider;data_provider"  --add-data="layers;layers"  --add-data="models;models"  --add-data="sample_original_data;sample_original_data"  --add-data="utils;utils" --add-data="run.py;." --add-data="sample_original_data/produce_data_from_bin.py;." sample_original_data/produce_data_from_bin_GUI.py -F -p C:\\Users\\12919\\anaconda3\\envs\\DLpytorch\\lib\\site-packages

pyinstaller --onedir --windowed --add-data="convert_paths;convert_paths" --add-data="data_provider;data_provider"  --add-data="layers;layers"  --add-data="models;models"  --add-data="sample_original_data;sample_original_data"  --add-data="utils;utils" --add-data="run.py;." --add-data="sample_original_data/produce_data_from_bin.py;." --add-data="C:\\Users\\12919\\anaconda3\\envs\\DLpytorch\\lib\\site-packages;.[注意：这里要替换成开发时虚拟环境的地址]" -F runGUI.py 
```


下采样功能在sample_original_data里，为了便于使用也打包成produce_data_from_bin_GUI.exe了，因为功能相对简单，这个exe可以直接拖到其他地方使用。