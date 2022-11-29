# GPGPU

### Authors

* Alexandre Lemonnier
* Victor Simonin
* Adrien Barens
* Sarah Gutierez

## Usage

### Build

For building the project, you need to have [CMake](https://cmake.org/) installed.
In `cpu_implem_opencv`, `cpu_implem` and `gpu_implem` directories, run the following commands:
```bash
$ cmake -B build
$ cd build
$ make
```

Then, for running the program in `cpu_implem` and `gpu_implem`, you can use the following command:
```bash
$ ./program_name [--save] <image_ref_path> <image_test_path> [image_test_path2] [image_test_path3] ...
```

### Generate frames

To generate frames from a video, you can use the script in the `tools` directory:
```bash
$ python3 frame.py <video_name> <video_path>
```

### Visualize

For visualizing the bounding boxes from the output json, you can use the python script `visualize.py` in the tools directory.
```bash
$ cat output.json | python visualize.py
```

You can choose to run one of the implementation in the `object_detection.py` python script on an image folder.
```bash
$ python object_detection.py <image_folder_path>
```

### Benchmark

To generate the benchmark, go into the `benchmarks` directory and run the following command:
```bash
$ cmake -B build
$ cd build
$ make
$ ./benchmark
```
or with
```bash
$ python python/benchmark.py
```