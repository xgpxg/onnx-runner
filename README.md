## onnx-runner

<img src="https://img.shields.io/badge/ONNX RUNNER-0.1.1-darkgreen?"  alt="ONNX RUNNER"/> <img src="https://img.shields.io/badge/ORT-2.0.0%20RC.8-darkgreen?link=https%3A%2F%2Fgithub.com%2Fpykeio%2Fort"  alt="ORT"/> <img src="https://img.shields.io/badge/ONNXRUNTIME-1.19.2-darkgreen?"  alt="ONNXRUNTIME"/> <img src="https://img.shields.io/badge/OPENCV-4.10.0-darkgreen?"  alt="OPENCV"/>

![build workflow](https://github.com/xgpxg/onnx-runner/actions/workflows/build.yml/badge.svg)
![release workflow](https://github.com/xgpxg/onnx-runner/actions/workflows/release.yml/badge.svg)

Use [ORT](https://github.com/pykeio/ort) to run ONNX model.

Currently, only YOLO models are supported, and other ONNX models may be supported in the
future

## Install

### Requirements

- If you want to use CPU to run onnx-runner, nothing to install
- If you want to use GPU to run onnx-runner, you need install CUDA 12.x and CUDNN 9.x

### Windows

- Download latest
  version: [onnx-runner-0.1.1-windows.tar.gz](https://github.com/xgpxg/onnx-runner/releases/download/v0.1.1/onnx-runner-v0.1.1-windows.tar.gz)
- Or download from release page: [Releases](https://github.com/xgpxg/onnx-runner/releases)

- Extract `onnx-runner-{version}-windows.tar.gz` to your path. The compressed package already includes the necessary
  dependencies for running ONNX and OpenCV. You don't need to download any other dependencies

- Run onnx-runner `` with CMD or PowerShell

  ```shell
  onnx-runner.exe  -m <your_onnx_model> -i <your_input> --show
  ```

### Ubuntu

- Download and install

  ```shell
  # Download latest package
  wget https://github.com/xgpxg/onnx-runner/releases/download/v0.1.1/onnx-runner_0.1.1_amd64.deb
  
  # Install package
  sudo apt -f install ./onnx-runner_0.1.1_amd64.deb
  ```

  Note：The OpenCV will be installed by default


- Run onnx-runner

  ```shell
  onnx-runner -m <your_onnx_model> -i <your_input> --show
  ```

### Other Linux

- Download latest
  version: [onnx-runner-v0.1.1-linux.tar.gz](https://github.com/xgpxg/onnx-runner/releases/download/v0.1.1/onnx-runner-v0.1.1-linux.tar.gz)

- Extract `onnx-runner-{version}-linux.tar.gz` to your path.

- Copy `libonnxruntime.so` to /usr/lib

- Install `Opencv`

- Run onnx-runner

  ```shell
  onnx-runner -m <your_onnx_model> -i <your_input> --show
  ```

### MacOS

Not currently supported

## usage

### CLI

```shell
onnx-runner -m yolov8n.onnx -i image.jpg --show
```

For more information, see help:

```shell
onnx-runner -h

Usage: onnx-runner.exe [OPTIONS] --model <MODEL> --input <INPUT>

Options:
  -m, --model <MODEL>                YOLO onnx model file path, support version: v5, v7, v8, v10, and v11
  -i, --input <INPUT>                Input source, like image file, http image, camera, or rtsp
      --yolo-version <YOLO_VERSION>  The number of YOLO version, like 5, 7 ,8 ,10, or 11. Specifically, for YOLO 10, it needs to be set up [default: 8]
      --show                         Should the detection results be displayed in the gui window, default is false
  -h, --help                         Print help
  -V, --version                      Print version

```

Supported input sources:

| Input               | Example                                                                |
|---------------------|------------------------------------------------------------------------|
| Local image file    | D:/images/img.png                                                      |
| Internet image file | https://cdn.pixabay.com/photo/2019/11/05/01/00/couple-4602505_1280.jpg |
| Local video file    | D:/images/video.mp4                                                    |
| Internet video file | https://cdn.pixabay.com/video/2024/06/04/215258_large.mp4              |
| Local camera        | camera://0                                                             |
| Ip camera(RTSP)     | rtsp://192.168.1.5:554                                                 |

### Lib

You need to install `rust` and `cargo`, then add onnx-runner to your project.

```shell
cargo add onnx-runner
```

Example

```rust
fn main() {
    //Use default config
    let mut config = ModelRunConfig::default();
    //Create a new runner
    let runner = ModelRunner::new(args.model.as_str(), config).unwrap();
    //Run with input. The input can be a local image, a network image, a camera, or a remote camera that supports RTSP
    runner.run(args.input.as_str(), ModelRunner::no_pre, |res, mut mat| {
        //Your code in here. You can send result to a http 
        println!("Result: {:?}", &res);
    },
    )?;
}
```

## CPU/GPU supports

All CPU are supported.

Currently only supports Nvidia GPUs. You need install CUDA 12.x + and cudnn 9.x + on your device.

## Troubleshooting

- I have installed CUDA and CUDNN, but why is the CPU still used instead of the GPU?

  First check whether the CUDA environment variables have been configured, and then check whether the CUDNN dependency
  libraries have been copied to the CUDA directory. Pay attention to the versions of CUDA and CUDNN. Currently only
  CUDA12.x and CUDNN9.x are supported.