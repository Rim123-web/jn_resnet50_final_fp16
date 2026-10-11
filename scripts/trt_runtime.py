from __future__ import annotations

import ctypes
import ctypes.util
import sys
from pathlib import Path

import numpy as np


USER_SITE = "/home/jetson/.local/lib/python3.8/site-packages"


def import_tensorrt():
    if USER_SITE not in sys.path:
        sys.path.insert(0, USER_SITE)
    import tensorrt as trt
    return trt


class TensorRTRunner:
    def __init__(self, engine_path: str | Path):
        self.trt = import_tensorrt()
        self.cudart, self.cudart_path = self._load_cudart()
        self._configure_cudart()
        self.stream = ctypes.c_void_p()
        self._check(self.cudart.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        self.events: list[ctypes.c_void_p] = []
        self.allocations: list[ctypes.c_void_p] = []
        logger = self.trt.Logger(self.trt.Logger.WARNING)
        runtime = self.trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT execution context creation failed")
        self.bindings = [0] * self.engine.num_bindings
        self.binding_info = []
        self.input_index = None
        self.outputs = {}
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            is_input = bool(self.engine.binding_is_input(index))
            shape = tuple(int(value) for value in self.engine.get_binding_shape(index))
            if is_input and any(value < 0 for value in shape):
                self.context.set_binding_shape(index, (1, 3, 640, 640))
            runtime_shape = tuple(int(value) for value in self.context.get_binding_shape(index))
            dtype = self.engine.get_binding_dtype(index)
            self.binding_info.append(
                {
                    "index": index,
                    "name": name,
                    "is_input": is_input,
                    "engine_shape": list(shape),
                    "runtime_shape": list(runtime_shape),
                    "dtype": str(dtype),
                }
            )
            host_dtype = self.trt.nptype(dtype)
            nbytes = int(np.prod(runtime_shape)) * np.dtype(host_dtype).itemsize
            pointer = ctypes.c_void_p()
            self._check(self.cudart.cudaMalloc(ctypes.byref(pointer), nbytes), "cudaMalloc")
            self.allocations.append(pointer)
            self.bindings[index] = int(pointer.value)
            if is_input:
                self.input_index = index
                self.input_shape = runtime_shape
                self.input_dtype = host_dtype
                self.input_device = pointer
            else:
                self.outputs[index] = np.empty(runtime_shape, dtype=host_dtype)
        if self.input_index is None:
            raise RuntimeError("TensorRT engine has no input binding")

    @staticmethod
    def _load_cudart():
        candidates = [
            ctypes.util.find_library("cudart"),
            "libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return ctypes.CDLL(candidate), candidate
            except OSError:
                pass
        raise RuntimeError("CUDA Runtime library not found")

    def _configure_cudart(self):
        c = self.cudart
        c.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        c.cudaMalloc.restype = ctypes.c_int
        c.cudaFree.argtypes = [ctypes.c_void_p]
        c.cudaFree.restype = ctypes.c_int
        c.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        c.cudaMemcpyAsync.restype = ctypes.c_int
        c.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        c.cudaStreamCreate.restype = ctypes.c_int
        c.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        c.cudaStreamSynchronize.restype = ctypes.c_int
        c.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        c.cudaStreamDestroy.restype = ctypes.c_int
        c.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        c.cudaEventCreate.restype = ctypes.c_int
        c.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        c.cudaEventRecord.restype = ctypes.c_int
        c.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        c.cudaEventSynchronize.restype = ctypes.c_int
        c.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        c.cudaEventElapsedTime.restype = ctypes.c_int
        c.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        c.cudaEventDestroy.restype = ctypes.c_int
        c.cudaDeviceSynchronize.argtypes = []
        c.cudaDeviceSynchronize.restype = ctypes.c_int

    @staticmethod
    def _check(code: int, operation: str):
        if code != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {code}")

    def _event_pair(self):
        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        self._check(self.cudart.cudaEventCreate(ctypes.byref(start)), "cudaEventCreate")
        self._check(self.cudart.cudaEventCreate(ctypes.byref(stop)), "cudaEventCreate")
        self.events.extend([start, stop])
        return start, stop

    def _elapsed(self, start, stop) -> float:
        elapsed = ctypes.c_float()
        self._check(self.cudart.cudaEventSynchronize(stop), "cudaEventSynchronize")
        self._check(
            self.cudart.cudaEventElapsedTime(ctypes.byref(elapsed), start, stop),
            "cudaEventElapsedTime",
        )
        value = float(elapsed.value)
        self.cudart.cudaEventDestroy(start)
        self.cudart.cudaEventDestroy(stop)
        self.events = [event for event in self.events if event not in (start, stop)]
        return value

    def infer(self, input_array: np.ndarray, timed: bool = True):
        host = np.ascontiguousarray(input_array.astype(self.input_dtype, copy=False))
        if tuple(host.shape) != tuple(self.input_shape):
            raise ValueError(f"Input shape {host.shape} != {self.input_shape}")

        h2d_start, h2d_stop = self._event_pair()
        self._check(self.cudart.cudaEventRecord(h2d_start, self.stream), "cudaEventRecord")
        self._check(
            self.cudart.cudaMemcpyAsync(
                self.input_device,
                ctypes.c_void_p(host.ctypes.data),
                host.nbytes,
                1,
                self.stream,
            ),
            "cudaMemcpyAsync H2D",
        )
        self._check(self.cudart.cudaEventRecord(h2d_stop, self.stream), "cudaEventRecord")
        h2d_ms = self._elapsed(h2d_start, h2d_stop)

        infer_start, infer_stop = self._event_pair()
        self._check(self.cudart.cudaEventRecord(infer_start, self.stream), "cudaEventRecord")
        ok = self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=int(self.stream.value)
        )
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 returned False")
        self._check(self.cudart.cudaEventRecord(infer_stop, self.stream), "cudaEventRecord")
        inference_ms = self._elapsed(infer_start, infer_stop)

        d2h_start, d2h_stop = self._event_pair()
        self._check(self.cudart.cudaEventRecord(d2h_start, self.stream), "cudaEventRecord")
        for index, output in self.outputs.items():
            self._check(
                self.cudart.cudaMemcpyAsync(
                    ctypes.c_void_p(output.ctypes.data),
                    ctypes.c_void_p(self.bindings[index]),
                    output.nbytes,
                    2,
                    self.stream,
                ),
                "cudaMemcpyAsync D2H",
            )
        self._check(self.cudart.cudaEventRecord(d2h_stop, self.stream), "cudaEventRecord")
        d2h_ms = self._elapsed(d2h_start, d2h_stop)
        self._check(self.cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        output = next(iter(self.outputs.values())).copy()
        return output, {
            "h2d_ms": h2d_ms,
            "inference_ms": inference_ms,
            "d2h_ms": d2h_ms,
            "transfer_inclusive_ms": h2d_ms + inference_ms + d2h_ms,
        }

    def synchronize(self):
        self._check(self.cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def close(self):
        for event in self.events:
            if event and event.value:
                self.cudart.cudaEventDestroy(event)
        self.events.clear()
        for pointer in self.allocations:
            if pointer and pointer.value:
                self.cudart.cudaFree(pointer)
        self.allocations.clear()
        if self.stream and self.stream.value:
            self.cudart.cudaStreamDestroy(self.stream)
            self.stream = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
