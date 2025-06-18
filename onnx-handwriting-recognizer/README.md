If you want to use **ONNX Runtime in a C++ project** using **CMake and NuGet**, the easiest way is to integrate the **ONNX Runtime NuGet package** directly into your CMake project. This avoids building ONNX Runtime from source and works well for both 32-bit and 64-bit targets.

------

## ✅ Steps to Use ONNX Runtime with CMake + NuGet

------

### 🔧 1. **Install NuGet (if not already)**

If `nuget.exe` is not available, download it from:
 🔗 https://www.nuget.org/downloads

Place it somewhere in your system `PATH` or in your project folder.

------

### 📦 2. **Download ONNX Runtime NuGet Package**

You can use the following command:

```bash
nuget install Microsoft.ML.OnnxRuntime -Version 1.17.0 -OutputDirectory external
```

Or run `build.sh`

You will get:

```
external/
└── Microsoft.ML.OnnxRuntime.1.17.0/
    └── build/
    └── runtimes/
    └── lib/native/
    └── include/
```

------

### 🧪 3. **Test Your Build**

Compile and run:

```cpp
mkdir build
cd build
cmake ..
```

